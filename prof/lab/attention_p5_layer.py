"""Whole-layer paired A/B on the host's (S10's) prepared state, identical public inputs, judge-style graph replays.

Variants (each its own CUDA graph of kernel_fn on the same static inputs; every function of the host module rebound
into a copy of its globals with the overrides below, so the host itself is never changed):
  A   the host as prepared;  A2 a second capture of A (A:A noise)
  F   small M (t <= 32): S8's one-LOP3 NT1 (moe_fused<1>, nt1_call(.., 'lop3', 2, 264)) on the same recoded bank
      instead of the flag-templated cell (95 at M=1, 69 at M=8/16/32)
  N   BF16 qkv front off: _attn_sumcast is the identity, so attn_main rows sum + cast with torch and the layer ext
      reads the FP32 splits, as in S8
  FN  both (S8's small-M layer)
  Qc/Ql/Qb  verify 1 only: the qkv projection forced to custom GEMV / cuBLASLt algo 2 / F.linear (the prepare tuner
      picked ('lt', 2) in S8's run and 'custom' in S9's)
Mechanism contexts at t <= 32 (static inputs from the seed's own prefix): RM = route + MoE graph, M = MoE alone,
QAO = qkv projection + attention + O.
Each seed: copy the fixture into the static inputs, replay every graph once and gate it against A (route IDs exact
up to 2% near-tie flips, relative L2 <= 1%; a failing cell is reported, still timed), then AB_ROUNDS rounds, each
replaying every graph once in a rotating order with lab_cold_l2() (1 GiB read) before each replay.
Lines: L <shape> <variant> r=<mean of per-seed time ratios vs A> se=<std error> d=<mean us delta> A=<A us> n=<seeds>
[gate]; LOAD <load> <variant> r=<46 sliding + 16 full weighted time ratio vs A>; C <shape> <context> ...;
P <shape> <plan>.
"""
import gc
import math
import statistics
import time
import types
import torch
from torch.utils.cpp_extension import load_inline

AB_NT1_CUDA = ''
AB_NT1_CPP = ''
AB_SEEDS = tuple(172903 + 7919 * i for i in range(16))
AB_ROUNDS = 8
AB_CASES = ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0), (32, 1, 2048), (32, 1, 0),
            (16, 8, 2048), (16, 8, 0), (32, 8, 2048), (32, 8, 0))
AB_LOADS = {(1, 1): 'decode 1', (16, 1): 'decode 16', (32, 1): 'decode 32', (1, 8): 'verify 1', (16, 8): 'verify 16',
            (32, 8): 'verify 32'}
_AB_NT1 = []


def ab_label(key):
    b, q, w = key
    return ('d' if q == 1 else 'v') + str(b) + ('S' if w else 'F')


def ab_nt1_module():
    if not _AB_NT1:
        import hashlib
        tag = hashlib.sha256(AB_NT1_CUDA.encode()).hexdigest()[:12]
        target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
        mod = load_inline(name='muse_nt1_limit_lop3' + tag, cpp_sources=AB_NT1_CPP, cuda_sources=AB_NT1_CUDA,
                          functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], no_implicit_headers=True,
                          with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        _AB_NT1.append(mod)
    return _AB_NT1[0]


def ab_fused(data, st, flags, stages=2, grid=264):
    """S8's nt1_call(data, s4_nt1_bank, 'lop3', 2, 264) on the same bank (flags ignored)."""
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]; p = m * 8; dev = x.device
    mod = ab_nt1_module()
    if mod.smem_needed(1, 2, m) > mod.smem_optin: raise ValueError('lop3 stages2 shared memory exceeds device capacity')
    grid = min(264, st['sms'] * mod.resident_blocks(1, 2, m))
    bank = st['banks']['lop3']
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev); ss = torch.empty((p,), device=dev); out = torch.empty_like(x)
    cnt = st.setdefault('ab_lop3_counter', torch.zeros(64, device=dev, dtype=torch.int32))
    tensors = [x, ids, tw, st['pnw'], bank['w13p'], bank['w2p'], st['gs13'], st['gs2'], inter, y, ss, cnt, out]
    mod.run([v.data_ptr() for v in tensors], m, data['config']['local_start'], 2, grid, torch.cuda.current_stream(dev).cuda_stream)
    return out


def ab_namespace(overrides):
    """Every plain function of the host module rebound into a copy of its globals, then the overrides."""
    g = kernel_fn.__globals__
    v = dict(g)
    for k, f in g.items():
        if isinstance(f, types.FunctionType) and f.__globals__ is g:
            h = types.FunctionType(f.__code__, v, f.__name__, f.__defaults__, f.__closure__)
            h.__kwdefaults__ = f.__kwdefaults__
            h.__dict__.update(f.__dict__)
            v[k] = h
    v.update(overrides)
    return v


def ab_copy(dst, src):
    if isinstance(dst, torch.Tensor):
        if dst.shape != src.shape:raise RuntimeError(f'fixture shape {tuple(src.shape)} != {tuple(dst.shape)}')
        dst.copy_(src);return
    if isinstance(dst, dict):
        for k in dst:ab_copy(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):ab_copy(a, b)


def ab_graph(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out = fn()
    graph.replay();torch.cuda.synchronize()
    return graph, out


def ab_tensors(out):
    if isinstance(out, torch.Tensor):return [out]
    if isinstance(out, (list, tuple)):return [t for o in out for t in ab_tensors(o)]
    return []


def ab_gate(got, want):
    """'' if close; else a short reason. Near-tie route flips (<= 2% of tokens) leave the float comparison."""
    got, want = ab_tensors(got), ab_tensors(want)
    flipped = None
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            rows = (a.reshape(a.shape[0], -1).sort(-1).values != b.reshape(b.shape[0], -1).sort(-1).values).any(-1)
            n = int(rows.sum())
            if n > max(1, a.shape[0] // 50):return f'ids{n}/{a.shape[0]}'
            flipped = rows if flipped is None else flipped | rows
    worst = 0.
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):continue
        if flipped is not None and a.shape[0] == flipped.shape[0] and bool(flipped.any()):a, b = a[~flipped], b[~flipped]
        if a.numel() == 0:continue
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()):return 'nonfinite'
        worst = max(worst, float((af - bf).norm() / bf.norm().clamp_min(1e-12)))
    return f'L2={worst:.4f}' if worst > .01 else ''


def ab_time(graphs, rounds):
    """name -> per-round us; every round replays each graph once in a rotating order, cold L2 before each."""
    names = list(graphs);times = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:order = order[::-1]
        evs = []
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lab_cold_l2();s.record();graphs[n][0].replay();e.record();evs.append((n, s, e))
        torch.cuda.synchronize()
        for n, s, e in evs:times[n].append(s.elapsed_time(e) * 1000.)
    return times


def ab_summary(per_seed, base='A'):
    """per_seed: [{name: mean us}] -> {name: (mean ratio, se, mean delta us, A us, n)}."""
    out = {}
    for n in per_seed[0]:
        if n == base:continue
        rs = [s[n] / s[base] for s in per_seed if n in s];ds = [s[n] - s[base] for s in per_seed if n in s]
        se = statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else float('nan')
        out[n] = (statistics.mean(rs), se, statistics.mean(ds), statistics.mean(s[base] for s in per_seed), len(rs))
    return out


def ab_prefix(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    a = v['_best7_project'](attn, 'o', state, plan)
    h, _ = v['_best7_boundary'](a, data['stream'], state, plan, 'b1')
    return v['_best7_project'](h, 'pool_router', state, plan, out_dtype=torch.float32)


def ab_rm(v, pr, data, state, plan):
    x, _, wts, lids = v['_best7_route'](pr, state, data, plan)
    return v['_best7_expert'](pr, x, wts, lids, state, plan)


def ab_qao(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    return v['_best7_project'](attn, 'o', state, plan)


def ab_plan_line(key, state):
    plan = state['best7_plans'][key];t = key[0] * key[1]
    roles = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice' if t <= 8 else 'unpooler')
    proj = {r: plan['projection'].get(r, 'auto') for r in roles}
    proj = {r: state['proj_plan'].get((t, r), 'custom') if c == 'auto' else c for r, c in proj.items()}
    attn = plan['attention'] if plan['attention'] != 'auto' else state['attention_plan'].get((key[0], key[1], bool(key[2])), ('layer', None))
    short = lambda c: c if isinstance(c, str) else '/'.join(map(str, c))
    return (f'P {ab_label(key)} moe={plan["moe"]} attn={short(attn)} glue={plan["glue"]} '
            + ' '.join(f'{r[:4]}={short(c)}' for r, c in proj.items()))



P5_CONTROL_FUNCTION = "def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n    vq8=state.get('vq8_rows',{}).get((cfg['batch'],cfg['q_len'],bool(cfg['window'])))\n    if vq8 is not None:return _vq8_attention(qkv,data,vq8)\n    if choice=='auto':choice=state['attention_plan'].get((cfg['batch'],cfg['q_len'],bool(cfg['window'])),('layer',None))\n    # This architecture path keeps the ordinary screen portable. The forced\n    # Hopper screen executes the same FA3 Python calls with a portable launch.\n    if choice[0]!='layer' and not _attn_port_hopper(qkv[0].device):return _layer_stage_attention(qkv,data,state)\n    if plan['glue']=='best5' and (choice[0]=='fa3' or (choice[0]=='layer' and cfg['q_len']==8 and _attn_port_hopper(qkv[0].device))):\n        value=qkv\n        if choice[0]=='fa3':value=(_attn_port_raw(qkv,data).reshape(-1,QKV_SIZE).unsqueeze(0),1)\n        q,gate=_best7_old_qk_prep(value,data,state,q_fp8=True)\n        return _attention_fa3_public(q,gate,data,state) if choice[0]=='layer' else _fa3_call(q,gate,data,choice[1])\n    return _attn_port_variant(qkv,data,state,choice)"
"""Whole-layer paired P5 port vs frozen final S11. A2 is a second identical control graph."""
def lab(deadline,weights,state):
    started=time.monotonic();dev=weights['qkv_gate'].device
    lines=['META final S11 whole-layer paired direct-partials port; 16 seeds x8 rounds, rotating graph order, read-flushed L2; ratio=variant/S11 (<1 wins)',
           'META A2=S11 duplicate, PP=direct-partials port with exact sincos; same static inputs/state/projection and MoE plans, TASK host correctness plus diagnostic A/B output check']
    # The exact original function from frozen final S11, bound to the control's own globals.
    a=ab_namespace({})
    exec(compile(P5_CONTROL_FUNCTION,'<p5_s11_attention>','exec'),a)
    b=ab_namespace({})
    results={};incomplete=[]
    for key in AB_CASES:
        if time.monotonic()>deadline-8:incomplete.append(ab_label(key));continue
        B,L,w=key
        if L!=1:continue
        lines.append(ab_plan_line(key,state))
        def mk(seed):
            return _ap_generate_input(batch=B,ctx=80000,layer='sliding' if w else 'full',q_len=L,
                spread='agentic' if B>1 else 'fixed',dist='soft',page=1,seed=seed,device=dev)
        data=mk(AB_SEEDS[0]);graphs={};gates={}
        for name,v in [('A',a),('A2',a),('PP',b)]:
            try:graphs[name]=ab_graph(lambda v=v:v['kernel_fn'](data,state))
            except Exception as e:gates[name]='capture:'+type(e).__name__+':'+str(e)[:100]
        if not all(n in graphs for n in ('A','A2','PP')):
            lines.append('CAPTURE '+ab_label(key)+' '+str(gates));continue
        seeds=[]
        for seed in AB_SEEDS:
            if time.monotonic()>deadline-4:incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed!=AB_SEEDS[0]:ab_copy(data,mk(seed))
            for g,_ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for name,(_,out) in graphs.items():
                if name!='A' and (reason:=ab_gate(out,graphs['A'][1])):gates.setdefault(name,reason)
            times=ab_time(graphs,AB_ROUNDS)
            seeds.append({n:statistics.mean(v) for n,v in times.items()})
        if seeds:results[key]=(ab_summary(seeds),gates)
        for n,(r,se,d,aus,k) in sorted(results.get(key,({},{}))[0].items()):
            lines.append(f'L {ab_label(key)} {n} r={r:.5f} se={se:.5f} d={d:+.2f} A={aus:.2f} n={k} '+gates.get(n,'PASS'))
        graphs=None;gc.collect();torch.cuda.empty_cache()
    for (B,L),label in AB_LOADS.items():
        keys=[(B,L,2048),(B,L,0)]
        if not all(k in results for k in keys):continue
        for name in ('A2','PP'):
            rs=[results[k][0][name] for k in keys]
            den=sum(w*r[3] for w,r in zip((46,16),rs));num=sum(w*r[3]*r[0] for w,r in zip((46,16),rs))
            lines.append(f'LOAD {label} {name} r={num/den:.5f}')
    if incomplete:lines.append('INCOMPLETE '+' '.join(incomplete))
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
