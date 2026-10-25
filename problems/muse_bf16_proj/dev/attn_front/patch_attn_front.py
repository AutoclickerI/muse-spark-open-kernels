"""attn_main reads the qkv projection's FP32 split partials and launches as a PDL dependent.

    python3 problems/muse_bf16_proj/dev/attn_front/patch_attn_front.py <cand.py> <out.py>

S6's attn_main path (decode 1, decode 16/32 full) converts the projection's (P, S) partials with
`partial.sum(0).to(bfloat16)`: two torch kernels (H200 fecc29af: reduce 3.7 us + elementwise 1.0 us at
decode 1, 4.2 + 1.3 at decode 16 full), without PDL, on the critical path. The 'layer' attention ext
already reads partials (qkv_pair); this gives the port's attn_main the same ABI, sums the splits with
all loads in flight, and rounds to BF16 exactly where the torch path did. attn_main is also launched
with programmatic stream serialization: page-table reads, positions, RoPE angles and the first
STAGES-1 K/V cache tiles go out before `griddepcontrol.wait`; every read of the projection's output
comes after it. Before the wait the kernel only reads inputs no in-flight kernel writes, and it
writes nothing.
"""
import hashlib
import sys
from pathlib import Path


def _sub(body, old, new, count=1):
    if body.count(old) != count:
        raise SystemExit(f'expected {count} of {old[:70]!r}, found {body.count(old)}')
    return body.replace(old, new)


HELPER = r'''
// qkv projection output: BF16 [T, N] (qkv_float 0) or FP32 split partials [S, T, N] summed here
// (all S loads in flight) and rounded to BF16 as the torch reduction + cast did.
DEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {
  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);
  const float* q = static_cast<const float*>(p.qkv) + off;
  float a = 0.f, b = 0.f;
  for (int s0 = 0; s0 < p.qkv_splits; s0 += 8) {
    float2 z[8];
#pragma unroll
    for (int u = 0; u < 8; ++u)
      z[u] = (s0 + u < p.qkv_splits) ? *reinterpret_cast<const float2*>(q + (s0 + u) * p.qkv_ps) : make_float2(0.f, 0.f);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      a += z[u].x;
      b += z[u].y;
    }
  }
  return __floats2bfloat162_rn(a, b);
}
'''

OLD_PROLOGUE = r'''  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                      (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  long long nloc[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = *reinterpret_cast<const __nv_bfloat162*>(p.qkv + (long long)(b * L + j) * p.qkv_stride +
                                                        (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
      nloc[i] = p.cache_loc[b * L + j];
    }
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;

  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each, with tile 0.
  {
    const int rank = split % p.cs;
    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;
    for (int i = tid; i < nrows * 8; i += NTHR) {
      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;
      cp_async16(smem_u32(sG + k * D + c * 8),
                 p.qkv + (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8, 16);
    }
  }
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
'''

NEW_PROLOGUE = r'''  constexpr int QR = 4 * L;  // query rows per warp: r = warp + 4 i -> (token r / 16, head r % 16)
  long long nloc[L];  // new rows per warp: r = warp + 4 i -> (token r / 4, head (r / 2) % 2, k|v r % 2)
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) nloc[i] = p.cache_loc[b * L + ((warp + 4 * i) >> 2)];
  }
  constexpr int CSN = (L * 32 + NTHR - 1) / NTHR;
  float ang[CSN];
#pragma unroll
  for (int c = 0; c < CSN; ++c) {
    const int i = tid + c * NTHR;
    ang[c] = 0.f;
    if (p.inv_freq && i < L * 32) ang[c] = static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31];
  }
  const float ksc = *p.k_scale, vsc = *p.v_scale;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) issue(s, rows[s]);
    cp_commit();
  }
  // PDL: everything above reads only inputs the qkv projection does not write. From here on the
  // projection's output is read (and the cache is written further down).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __nv_bfloat162 qv[QR];
#pragma unroll
  for (int i = 0; i < QR; ++i) {
    const int r = warp + 4 * i, j = r >> 4, hh = r & 15;
    qv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride + (kvg * 16 + hh) * D + 2 * lane);
  }
  __nv_bfloat162 nv[L];
  if (last) {
#pragma unroll
    for (int i = 0; i < L; ++i) {
      const int r = warp + 4 * i, j = r >> 2, hk = (r >> 1) & 1, isv = r & 1;
      nv[i] = qkv_pair(p, (long long)(b * L + j) * p.qkv_stride + (p.hq + (isv ? p.hkv : 0) + kvg * 2 + hk) * D + 2 * lane);
    }
  }
  // Gate rows this CTA finalizes (rows rank, rank + CS, ...): 8 x 16 bytes each (read at the end).
  {
    const int rank = split % p.cs;
    const int nrows = (L * 16 - rank + p.cs - 1) / p.cs;
    for (int i = tid; i < nrows * 8; i += NTHR) {
      const int k = i >> 3, c = i & 7, rr = rank + k * p.cs, j = rr >> 4, hh = rr & 15;
      const long long go = (long long)(b * L + j) * p.qkv_stride + (p.hq + 2 * p.hkv + kvg * 16 + hh) * D + c * 8;
      __nv_bfloat162 gq[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) gq[u] = qkv_pair(p, go + 2 * u);
      *reinterpret_cast<uint4*>(sG + k * D + c * 8) = *reinterpret_cast<const uint4*>(gq);
    }
  }
'''

OLD_LAUNCH = r'''  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = p.cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);'''

NEW_LAUNCH = r'''  cudaLaunchAttribute at[2];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = p.cs;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[1].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 2;
  return cudaLaunchKernelEx(&cfg, attn_main<L, STAGES>, p);'''

OLD_SIG = 'void attn_forward(int64_t qkv, int64_t positions,'
NEW_SIG = 'void attn_forward(int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions,'

OLD_FWD = '''def _attn_port_forward(raw, data):
    """Run the standalone four-stage q1 gather or its L=8 sibling."""
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']; device = raw.device
'''
NEW_FWD = '''def _attn_port_forward(qkv, data):
    """Run the standalone four-stage q1 gather or its L=8 sibling on the projection's (P, S) output."""
    partial, splits = qkv
    if partial.stride(-1) != 1 or (splits > 1 and partial.dtype != torch.float32):
        raise ValueError('attn_main reads BF16 [T, N] or FP32 split partials [S, T, N]')
    cfg = data['config']; batch, length = cfg['batch'], cfg['q_len']; device = partial.device
'''
OLD_CALL = "    ext.attn_forward(raw.data_ptr(), data['positions'].data_ptr(),"
NEW_CALL = ("    ext.attn_forward(partial.data_ptr(), splits, partial.stride(0), int(partial.dtype == torch.float32),\n"
            "           data['positions'].data_ptr(),")
OLD_STRIDE = "           raw.shape[-1], HQ, HKV, max_pages, ps, window, ns, cs, scale_log2,"
NEW_STRIDE = "           partial.stride(-2), HQ, HKV, max_pages, ps, window, ns, cs, scale_log2,"
OLD_VAR = '''    raw = _attn_port_raw(qkv, data)
    if label == 'fa3':
        return _attn_port_fa3(raw, data, setting)
    if label in ('attn_main', 'port_verify'):
        return _attn_port_forward(raw, data)
'''
NEW_VAR = '''    if label in ('attn_main', 'port_verify'):
        return _attn_port_forward(qkv, data)
    raw = _attn_port_raw(qkv, data)
    if label == 'fa3':
        return _attn_port_fa3(raw, data, setting)
'''


def patch_cuda(body: str, pdl: bool = True) -> str:
    """ATTN_PORT_CUDA_SRC text -> partial-reading (and, with pdl, PDL-launched) attn_main."""
    if 'qkv_pair' in body:
        raise SystemExit('attn_main already reads partials')
    body = _sub(body, '  const __nv_bfloat16* qkv;\n',
                '  const void* qkv;          // BF16 [T, N] or FP32 split partials [S, T, N]\n'
                '  long long qkv_ps;         // split stride in elements (FP32 partials)\n'
                '  int qkv_splits, qkv_float;\n')
    body = _sub(body, 'DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\n',
                'DEVI uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }\n' + HELPER)
    body = _sub(body, OLD_PROLOGUE, NEW_PROLOGUE)
    if pdl:
        body = _sub(body, OLD_LAUNCH, NEW_LAUNCH)
    body = _sub(body, OLD_SIG, NEW_SIG)
    body = _sub(body, '  p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);\n',
                '  p.qkv = reinterpret_cast<const void*>(qkv);\n  p.qkv_splits = (int)qkv_splits;\n'
                '  p.qkv_ps = qkv_ps;\n  p.qkv_float = (int)qkv_float;\n')
    return body


def patch_cpp(body: str) -> str:
    return _sub(body, OLD_SIG, NEW_SIG)


def _block(src, start_marker):
    start = src.index(start_marker)
    end = src.index('"""', start + len(start_marker))
    return start, end


def patch(src: str) -> str:
    s, e = _block(src, 'ATTN_PORT_CUDA_SRC = r"""')
    src = src[:s] + patch_cuda(src[s:e]) + src[e:]
    s, e = _block(src, 'ATTN_PORT_CPP_SRC = r"""')
    src = src[:s] + patch_cpp(src[s:e]) + src[e:]
    src = _sub(src, OLD_FWD, NEW_FWD)
    src = _sub(src, OLD_CALL, NEW_CALL)
    src = _sub(src, OLD_STRIDE, NEW_STRIDE)
    src = _sub(src, OLD_VAR, NEW_VAR)
    return src


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
