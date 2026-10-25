"""Fuse projection split reduction into final S11's single-launch attention port.
The generated source keeps its cluster gather and reduction and adds PDL waits before projection reads.
"""
import ast
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'problems/muse_decode_layer/cand_r3_best7_s11_final.py'


def literals():
    names = {'ATTN_PORT_CUDA_SRC', 'ATTN_PORT_CPP_SRC'}
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(BASE.read_text()).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in names}


def source(fast_rope=False):
    src = literals()['ATTN_PORT_CUDA_SRC']
    src = src.replace('  int qkv_stride;', '  int qkv_stride;\n  int proj_splits;\n  int gated;\n  int64_t split_stride;')
    helper = r'''
// Load all projection partials together, then round at the original BF16 boundary.
template <int N>
DEVI void proj_pairs(const Params& p, const int (&tk)[N], const int (&col)[N], __nv_bfloat162 (&out)[N]) {
  if (!p.proj_splits) {
    uint32_t raw[N];
#pragma unroll
    for (int i=0; i<N; i++) raw[i] = *(const uint32_t*)(p.qkv+(int64_t)tk[i]*p.qkv_stride+col[i]);
#pragma unroll
    for (int i=0; i<N; i++) memcpy(&out[i], &raw[i], 4);
  } else {
    float2 acc[N];
#pragma unroll
    for (int i=0; i<N; i++) acc[i]=make_float2(0.f,0.f);
    for (int s=0; s<p.proj_splits; s++) {
      float2 v[N];
#pragma unroll
      for (int i=0; i<N; i++) v[i]=*(const float2*)((const float*)p.qkv+(int64_t)s*p.split_stride+(int64_t)tk[i]*p.qkv_stride+col[i]);
#pragma unroll
      for (int i=0; i<N; i++) {acc[i].x+=v[i].x;acc[i].y+=v[i].y;}
    }
#pragma unroll
    for (int i=0; i<N; i++) out[i]=__floats2bfloat162_rn(acc[i].x,acc[i].y);
  }
}
'''
    src = src.replace('template <int L>\nstruct WarpState', helper+'\ntemplate <int L>\nstruct WarpState')
    # Historical KV gathers can run while the projection completes. No new slots are read.
    a = src.index('#pragma unroll\n  for (int s = 0; s < STAGES - 1; ++s) {', src.index('void __launch_bounds__'))
    b = src.index('\n  // RoPE angles', a)
    prefetch = src[a:b]
    src = src[:a] + src[b:]
    a = src.index('  constexpr int QR = 4 * L;')
    b = src.index('  constexpr int CSN', a)
    prep = r'''
  constexpr int QR = 4 * L;
  int toks[QR+L], cols[QR+L];
#pragma unroll
  for (int i=0; i<QR; i++) {
    const int r=warp+4*i, j=r>>4, hh=r&15;
    toks[i]=b*L+j; cols[i]=(kvg*16+hh)*D+2*lane;
  }
  long long nloc[L];
#pragma unroll
  for (int i=0; i<L; i++) {
    const int r=warp+4*i, j=r>>2, hk=(r>>1)&1, isv=r&1;
    toks[QR+i]=b*L+j; cols[QR+i]=(p.hq+(isv?p.hkv:0)+kvg*2+hk)*D+2*lane;
    nloc[i]=p.cache_loc[b*L+j];
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");
  __nv_bfloat162 qn[QR+L], qv[QR], nv[L];
  proj_pairs<QR+L>(p,toks,cols,qn);
#pragma unroll
  for (int i=0; i<QR; i++) qv[i]=qn[i];
#pragma unroll
  for (int i=0; i<L; i++) nv[i]=qn[QR+i];
'''
    src = src[:a] + prefetch + prep + src[b:]
    a = src.index('      cp_async16(smem_u32(sG')
    b = src.index('\n    }', a)
    src = src[:a] + r'''
      int gt[4], gc[4]; __nv_bfloat162 gv[4];
#pragma unroll
      for (int z=0; z<4; z++) {
        gt[z]=b*L+j;
        gc[z]=(p.hq+2*p.hkv+kvg*16+hh)*D+c*8+2*z;
      }
      proj_pairs<4>(p,gt,gc,gv);
#pragma unroll
      for (int z=0; z<4; z++) *(__nv_bfloat162*)(sG+k*D+c*8+2*z)=gv[z];''' + src[b:]
    src = src.replace('double scale_log2, int64_t stream)', 'double scale_log2, int64_t proj_splits, int64_t split_stride, int64_t qkv_width, int64_t stream)')
    src = src.replace('  p.qkv_stride = (int)qkv_stride;', '  p.qkv_stride = (int)qkv_stride;\n  p.proj_splits = (int)proj_splits;\n  p.split_stride = split_stride;')
    src = src.replace('  const bool gated = p.qkv_stride > (p.hq + 2 * p.hkv) * D;', '  const bool gated = p.gated;')
    src = src.replace('  p.out = reinterpret_cast<__nv_bfloat16*>(out);', '  p.gated = qkv_width > (hq + 2*hkv)*64;\n  p.out = reinterpret_cast<__nv_bfloat16*>(out);')
    # A PDL launch is safe because the consumer explicitly waits above; cached KV is independent.
    src = src.replace('  cudaLaunchAttribute at[1];', '  cudaLaunchAttribute at[2];', 1)
    src = src.replace('  cfg.numAttrs = 1;\n  return cudaLaunchKernelEx', '  at[1].id=cudaLaunchAttributeProgrammaticStreamSerialization;\n  at[1].val.programmaticStreamSerializationAllowed=1;\n  cfg.numAttrs = 2;\n  return cudaLaunchKernelEx', 1)
    if fast_rope:
        src = src.replace('sincosf(ang[c], &sn, &cs);', '__sincosf(ang[c], &sn, &cs);')
    return src


DRIVER = r'''
_PP_MOD = {}

def partial_port_ext(fast=False):
    if fast not in _PP_MOD:
        major, minor = torch.cuda.get_device_capability()
        arch='90a' if major==9 else f'{major}{minor}'
        cu=PARTIAL_FAST_CUDA if fast else PARTIAL_CUDA
        tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
        _PP_MOD[fast]=load_inline(name=f'attn_partial_{tag}_{arch}', cpp_sources=PARTIAL_CPP,
            cuda_sources=cu,functions=['attn_forward','max_cluster','occupancy'],
            extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    return _PP_MOD[fast]


def attention_partial_port(qkv, data, fast=False):
    P, S=qkv
    cfg=data['config']; B,L=cfg['batch'],cfg['q_len']; HQ_,HK_=cfg['num_q_heads'],cfg['num_kv_heads']
    dev=P.device; window=cfg['window']; ps=cfg['page_size']; pages=data['page_table'].shape[1]
    ext=partial_port_ext(fast)
    occ=(max(1,ext.occupancy(L)),{c:ext.max_cluster(L,c) for c in range(1,17)})
    ns,cs=_attn_main_plan(B,HK_//2,min(pages*ps,window) if window else pages*ps,L,window,_num_sms(dev),occ)
    rows=B*(HK_//2)*max(1,ns//cs)*L*16
    po=torch.empty((max(rows,1)*64,),device=dev,dtype=torch.float32)
    ml=torch.empty((max(rows,1)*2,),device=dev,dtype=torch.float32)
    counters=_attn_port_counters(B*(HK_//2)*cs,dev)
    out=torch.empty((B*L,HQ_*64),device=dev,dtype=torch.bfloat16)
    raw=P[0] if P.dtype==torch.bfloat16 and P.dim()==3 else P
    nsplit=S if P.dtype==torch.float32 else 0
    inv=_inv_freq(cfg['rope_theta'],dev) if window else None
    ext.attn_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),data['page_table'].data_ptr(),
        data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),data['k_scale'].data_ptr(),
        data['v_scale'].data_ptr(),inv.data_ptr() if inv is not None else 0,po.data_ptr(),ml.data_ptr(),counters.data_ptr(),
        po.data_ptr(),1,out.data_ptr(),B,L,raw.stride(-2),HQ_,HK_,pages,ps,window,ns,cs,
        cfg['sm_scale']*cfg['qk_scale_factor']*1.4426950408889634,nsplit,P.stride(0) if nsplit else 0,P.shape[-1],
        torch.cuda.current_stream(dev).cuda_stream)
    return out
'''


def cell():
    cpp = literals()['ATTN_PORT_CPP_SRC'].replace('double scale_log2, int64_t stream)',
        'double scale_log2, int64_t proj_splits, int64_t split_stride, int64_t qkv_width, int64_t stream)')
    return ('PARTIAL_CUDA = '+repr(source())+'\nPARTIAL_FAST_CUDA = '+repr(source(True))+'\nPARTIAL_CPP = '+repr(cpp)+'\n'+DRIVER)

if __name__ == '__main__':
    import hashlib
    p = Path(__file__).resolve().parent / 'out/partial_port.cu'
    s = source();p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
