"""Gate split for the q_len 1 rows: the attention output gate's projection leaves the qkv GEMV and runs beside
attention.

    python3 problems/muse_bf16_proj/dev/gate_split/patch_gate_split.py <cand.py> <out.py>

Needs `patch_gemv_bal` (the ('bal', ...) GEMV choices) and `patch_front_bf16` already in the candidate.

On a q_len 1 call in GATE_SPLIT_ROWS (decode 1 sliding/full, decode 16 sliding, decode 32 full) whose attention is
attn_main or the layer ext:
- the qkv projection reads only the q, k, v rows (state['qkv3'] = qkv_gate[:1280], a view): 44% fewer bytes
  before attention;
- attention runs without the gate. Both attention modules take "row wider than q + k + v" as "gate columns
  present", so a 1280-wide qkv means ungated output, with no ABI change;
- `_gate_split_kernel` (PDL) projects the gate rows (qkv_gate[1280:], a view) from the layer input before
  griddepcontrol.wait. Attention releases its dependents at entry: attn_main already does (S8), and the layer
  ext's attn_main and attn_combine do with this patch (attn_combine PDL-launched and waiting), so this kernel
  streams its 16.8 MB while attention runs. The last split CTA of each tile then waits, sums the partials in
  split order, and writes attn_out * sigmoid(gate) (BF16) for O. Its wait makes its completion imply
  attention's, so O's griddepcontrol.wait still covers the attention output.
- The partials and counters live in buffers allocated in prepare, never in the per-call pool. The kernel writes
  them before its wait, so they must not alias memory an earlier, still-running kernel reads.
FA3 rows, q_len 8 and the other rows are unchanged.

H200 chains (x -> qkv -> attention -> O -> b1, paired vs S10 + gemv_bal; prof/lab/gate_split.py, gate_sweep.py):
the gate kernel at S 4 (64 CTAs) beats S 8 / 16 and deeper pipelines; per layer b1 S -1.99 / -1.79 us, b1 F -0.64 /
-0.37, b16 S -0.77 / -2.23, b32 F -1.91 (9b7ffda0, 997bd383), b16 F +0.88 and b32 S +0.02 (left off). qkv3 saves
~3.5 us but the 16.8 MB gate GEMV outlasts attention's window and must finish before O (inferred).
"""
import hashlib
import sys
from pathlib import Path

QKV3 = 1280  # (HQ + 2 * HKV) * HD: q, k, v columns

# Attention modules: gate columns present iff the qkv row is wider than q + k + v.
PORT_OLD_STAGE = '''  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.
  {
'''
PORT_NEW_STAGE = '''  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.
  const bool gated = p.qkv_stride > (p.hq + 2 * p.hkv) * D;  // else ungated output (P7 gate split)
  if (gated) {
'''
PORT_OLD_FIN = '''    const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];
    const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
    const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
'''
PORT_NEW_FIN = '''    float g0 = 1.f, g1 = 1.f;
    if (gated) {
      const __nv_bfloat162 gt = reinterpret_cast<const __nv_bfloat162*>(sG + k * D)[lane];
      g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
      g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
    }
'''
LAYER_OLD_GATE = '''  const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);
  const float g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
  const float g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
'''
LAYER_NEW_GATE = '''  float g0 = 1.f, g1 = 1.f;
  if (p.qkv_stride > (p.hq + 2 * p.hkv) * D) {  // gate columns present (else P7 gate split: ungated output)
    const __nv_bfloat162 gt = qkv_pair(p, tok * p.qkv_stride + (p.hq + 2 * p.hkv + hg) * D + 2 * lane);
    g0 = 1.f / (1.f + __expf(-__bfloat162float(gt.x)));
    g1 = 1.f / (1.f + __expf(-__bfloat162float(gt.y)));
  }
'''
LAYER_OLD_MAIN = '''template <int L, int STAGES>
__global__ void __launch_bounds__(NTHR, 2) attn_main(const Params p) {
'''
LAYER_NEW_MAIN = LAYER_OLD_MAIN + '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // attn_combine launches early and waits
#endif
'''
LAYER_OLD_COMB = '''__global__ void __launch_bounds__(128) attn_combine(const Params p) {
'''
LAYER_NEW_COMB = LAYER_OLD_COMB + '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // dependents (the gate kernel, O) wait
  asm volatile("griddepcontrol.wait;" ::: "memory");               // attn_main's partials
#endif
'''
LAYER_OLD_LAUNCH = '''  attn_combine<L><<<(warps + 3) / 4, 128, 0, stream>>>(p);
'''
LAYER_NEW_LAUNCH = '''  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((warps + 3) / 4);
  cfg.blockDim = dim3(128);
  cfg.stream = stream;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, attn_combine<L>, p);
'''

KERNEL = '''

@triton.jit
def _gate_split_kernel(X, W, P, CNT, Y, G, T, x_stride, K, S, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       TB: tl.constexpr, NS: tl.constexpr):
    """Gate rows of qkv_gate, split-K, all before griddepcontrol.wait (reads only the layer input and W); the
    last split CTA of a tile waits (attention complete), sums the S partials in split order and writes
    attn_out * sigmoid(gate) as BF16 [T, N]."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    tl.extra.cuda.gdc_launch_dependents()
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    mask = t[None, :] < T
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=mask)
    tl.debug_barrier()
    arrived = tl.atomic_add(CNT + pn, 1, sem='acq_rel', scope='gpu')
    if arrived == S - 1:
        tl.extra.cuda.gdc_wait()
        tot = tl.zeros((BN, TB), tl.float32)
        for s in range(S):
            tot += tl.load(P + s * (T * N) + t[None, :] * N + n[:, None], mask=mask, other=0.0, cache_modifier='.cg')
        y = tl.load(Y + t[None, :] * N + n[:, None], mask=mask, other=0.0).to(tl.float32)
        tl.store(G + t[None, :] * N + n[:, None], (y * tl.sigmoid(tot)).to(tl.bfloat16), mask=mask)
        tl.atomic_xchg(CNT + pn, 0, sem='relaxed', scope='gpu')


GATE_SPLIT_CFG = (64, 4, 4)  # (BN, S, stages) of the gate GEMV: 16 x 4 = 64 CTAs beside attention
# (batch, sliding) rows that take the split: H200 9b7ffda0 + 997bd383 at S 4, us vs S10 + gemv_bal per layer:
# b1 S -1.99 / -1.79, b1 F -0.64 / -0.37, b16 S -0.77 / -2.23, b32 F -1.91; off: b16 F +0.88, b32 S +0.02.
GATE_SPLIT_ROWS = {(1, True), (1, False), (16, True), (32, False)}
GATE_SPLIT_QKV3 = ('bal', 64, 10, 4)  # q, k, v rows: 20 x 10 = 200 CTAs
_GATE_SPLIT_BUF = {}


def _gate_split_buffers(device, tokens):
    """Prepare-time (or first eager call) partials [S, T, 1024] and per-tile counters: never per-call memory."""
    buf = _GATE_SPLIT_BUF.get(tokens)
    if buf is None:
        BN, S, _ = GATE_SPLIT_CFG
        buf = _GATE_SPLIT_BUF[tokens] = (torch.empty((S, tokens, Q_SIZE), device=device, dtype=torch.float32),
                                         torch.zeros(Q_SIZE // BN, device=device, dtype=torch.int32))
    return buf


def _gate_split_prepare(state, device):
    state['qkv3'] = state['qkv_gate'][:QKV3_SIZE]
    state['gate_rows'] = state['qkv_gate'][QKV3_SIZE:]
    for t in (1, 8, 16, 32):
        _GEMV_CHOICES[(t, QKV3_SIZE, HIDDEN)] = GATE_SPLIT_QKV3
        _gate_split_buffers(device, t)


def _gate_split_apply(x, y, state):
    """attn_out [T, 1024] (ungated) -> attn_out * sigmoid(x @ gate_rows.T), one PDL kernel."""
    T, K = x.shape
    BN, S, NS = GATE_SPLIT_CFG
    P, cnt = _gate_split_buffers(x.device, T)
    g = torch.empty((T, Q_SIZE), device=x.device, dtype=torch.bfloat16)
    w = state['gate_rows']
    _gate_split_kernel[(Q_SIZE // BN, S)](x, w, P, cnt, y, g, T, x.stride(0), K, S, N=Q_SIZE, BN=BN, BK=128,
                                          TB=max(16, triton.next_power_of_2(T)), NS=NS, num_warps=4, launch_pdl=True)
    return g


def _gate_split_on(data, state, plan):
    cfg = data['config']
    if cfg['q_len'] != 1 or plan.get('dense_chain', False) or (cfg['batch'], bool(cfg['window'])) not in GATE_SPLIT_ROWS:
        return False
    choice = plan['attention']
    if choice == 'auto':
        choice = state['attention_plan'].get((cfg['batch'], cfg['q_len'], bool(cfg['window'])), ('layer', None))
    return choice[0] in ('attn_main', 'layer')
'''

OLD_RUN = '''    qkv=_best7_project(data['attn_input'],'qkv_gate',state,plan)
    if prefetch is not None:prefetch.begin('front')
    attn=_best7_attention(qkv,data,state,plan)
'''
NEW_RUN = '''    split=_gate_split_on(data,state,plan)
    qkv=_best7_project(data['attn_input'],'qkv3' if split else 'qkv_gate',state,plan)
    if prefetch is not None:prefetch.begin('front')
    attn=_best7_attention(qkv,data,state,plan)
    if split:attn=_gate_split_apply(data['attn_input'],attn,state)
'''
OLD_RAW = "    return raw.to(torch.bfloat16).view(cfg['batch'], cfg['q_len'], QKV_SIZE).contiguous()\n"
NEW_RAW = "    return raw.to(torch.bfloat16).view(cfg['batch'], cfg['q_len'], raw.shape[-1]).contiguous()\n"
OLD_CFG = '    (QKV_SIZE, HIDDEN): (64, 1024, 128),\n'
NEW_CFG = OLD_CFG + '    (QKV3_SIZE, HIDDEN): (64, 1024, 128),\n'
OLD_PREP = '    _best7_prepare_settings(state,weights,device)\n'
NEW_PREP = '    _gate_split_prepare(state,device)\n' + OLD_PREP


def _sub(src, old, new, count=1):
    if src.count(old) != count:
        raise SystemExit(f'expected {count} of {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def _patch_literal(src, name, subs):
    """Apply substitutions inside the string literal assigned to `name` (a top-level NAME = r\"\"\"...\"\"\" or '...')."""
    import ast
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == name:
            lines = src.splitlines(keepends=True)
            start = sum(len(x) for x in lines[:node.lineno - 1])
            end = sum(len(x) for x in lines[:node.end_lineno])
            value = ast.literal_eval(node.value)
            for old, new in subs:
                value = _sub(value, old, new)
            assert '"""' not in value and not value.endswith('\\')
            return src[:start] + f'{name} = r"""{value}"""\n' + src[end:]
    raise SystemExit(f'no {name}')


def patch(src: str) -> str:
    if '_gb_gemv' not in src or '_attn_sumcast_kernel' not in src:
        raise SystemExit('apply patch_gemv_bal and patch_front_bf16 first')
    src = _patch_literal(src, 'ATTN_PORT_CUDA_SRC', [(PORT_OLD_STAGE, PORT_NEW_STAGE), (PORT_OLD_FIN, PORT_NEW_FIN)])
    src = _patch_literal(src, 'CUDA_SRC', [(LAYER_OLD_GATE, LAYER_NEW_GATE), (LAYER_OLD_MAIN, LAYER_NEW_MAIN),
                                           (LAYER_OLD_COMB, LAYER_NEW_COMB), (LAYER_OLD_LAUNCH, LAYER_NEW_LAUNCH)])
    anchor = src.index('\n\n# (N, K) -> (BN, KS, BK)')
    src = src[:anchor] + KERNEL + src[anchor:]
    src = _sub(src, 'QKV_SIZE = ', f'QKV3_SIZE = {QKV3}  # q, k, v columns (P7 gate split)\nQKV_SIZE = ')
    for old, new in ((OLD_RUN, NEW_RUN), (OLD_RAW, NEW_RAW), (OLD_CFG, NEW_CFG), (OLD_PREP, NEW_PREP)):
        src = _sub(src, old, new)
    return src


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
