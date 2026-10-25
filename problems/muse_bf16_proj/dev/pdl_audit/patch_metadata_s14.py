"""S14 metadata preload patch; paired integration verification required."""
import ast,hashlib

BASE_SHA='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'

CUDA_HELPERS=r'''
#include <stdint.h>
// Timed-call inputs and immutable weights only. Never use for predecessor outputs.
__device__ __forceinline__ float pa_f32(const float* p) {
 float x;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(x):"l"(p):"memory");return x;
}
__device__ __forceinline__ int pa_i32(const int* p) {
 int x;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;
}
__device__ __forceinline__ long long pa_i64(const int64_t* p) {
 long long x;asm volatile("ld.global.ca.u64 %0,[%1];":"=l"(x):"l"(p):"memory");return x;
}
'''

TRITON_HELPERS='''
@triton.jit
def _pa_f32(p):
    x=tl.load(p)
    return tl.inline_asm_elementwise("mov.b32 $0,$1;", "=f,f", [x], dtype=tl.float32, is_pure=False, pack=1)
@triton.jit
def _pa_i32(p):
    return tl.inline_asm_elementwise("ld.global.ca.u32 $0,[$1];", "=r,l", [p], dtype=tl.int32, is_pure=False, pack=1)
@triton.jit
def _pa_i64(p):
    return tl.inline_asm_elementwise("ld.global.ca.u64 $0,[$1];", "=l,l", [p], dtype=tl.int64, is_pure=False, pack=1)
@triton.jit
def _pa_b16(p):
    x=tl.load(p)
    return tl.inline_asm_elementwise("mov.b16 $0,$1;", "=h,h", [x], dtype=tl.bfloat16, is_pure=False, pack=1)
@triton.jit
def _pa_keep(x):
    return tl.inline_asm_elementwise("mov.b32 $0,$1;", "=f,f", [x.to(tl.float32)], dtype=tl.float32, is_pure=False, pack=1)
@triton.jit
def _pa_keepi(x):
    return tl.inline_asm_elementwise("mov.b32 $0,$1;", "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1)
'''

def once(s,old,new):
    assert s.count(old)==1,(old[:90],s.count(old))
    return s.replace(old,new,1)

def glue_source(cuda):
    anchor='namespace cg = cooperative_groups;'
    helpers='''
// Ordinary const/restrict loads are sunk after ACQBULK by nvcc/ptxas. Inline
// PTX memory operations preserve their place before the dependency acquire.
__device__ __forceinline__ float4 gp_load4(const float* p) {
  float4 v;
  asm volatile("ld.global.ca.v4.f32 {%0,%1,%2,%3},[%4];"
      : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");
  return v;
}
__device__ __forceinline__ float gp_load1(const float* p) {
  float v;asm volatile("ld.global.ca.f32 %0,[%1];":"=f"(v):"l"(p):"memory");return v;
}
__device__ __forceinline__ int gp_loadi(const int* p) {
  int v;asm volatile("ld.global.ca.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
'''
    assert cuda.count(anchor)==1
    cuda=cuda.replace(anchor,helpers+'\n'+anchor)
    for field in ('alpha','beta','prew','postw'):
        old=f'*reinterpret_cast<const float4*>({field} + j)'
        assert old in cuda,(field,'missing')
        cuda=cuda.replace(old,f'gp_load4({field} + j)')
    old='*reinterpret_cast<const float4*>(prew + (rank * QT + q) * 4)'
    assert old in cuda
    cuda=cuda.replace(old,'gp_load4(prew + (rank * QT + q) * 4)')
    cuda=cuda.replace('tid < EP ? bias[e] : 0.f','tid < EP ? gp_load1(bias+e) : 0.f')
    cuda=cuda.replace('const int start = *lstart;','const int start = gp_loadi(lstart);')
    return cuda

def native(name,s):
    if name=='GLUE_CL_CUDA':return glue_source(s)
    if name=='CUDA_SRC':
        pos=s.index('__global__ void __launch_bounds__(128) attn_combine')
        a,b=s[:pos],s[pos:]
        b=once(b,'  const float vs = *p.v_scale;\n','')
        b=once(b,'attn_combine(const Params p) {','attn_combine(const Params p) {\n  const float vs = pa_f32(p.v_scale);')
        return CUDA_HELPERS+a+b
    if name=='PARTIAL_CUDA':
        a=s.index('  constexpr int CSN =');b=s.index('\n\n  // Gate rows',a)
        angles=s[a:b];s=s[:a]+s[b:]
        angles=angles.replace('static_cast<float>(p.positions[b * L + (i >> 5)]) * p.inv_freq[i & 31]',
                              'static_cast<float>(pa_i64(p.positions+b*L+(i>>5))) * pa_f32(p.inv_freq+(i&31))')
        angles=angles.replace('*p.k_scale','pa_f32(p.k_scale)').replace('*p.v_scale','pa_f32(p.v_scale)')
        s=once(s,'  asm volatile("griddepcontrol.wait;"',angles+'\n  asm volatile("griddepcontrol.wait;"')
        s=s.replace('p.seq_lens[b]','pa_i32(p.seq_lens+b)').replace('pt[t >> p.ps_log2]','pa_i32(pt+(t >> p.ps_log2))')
        s=s.replace('nloc[i]=p.cache_loc[b*L+j];','nloc[i]=pa_i64(p.cache_loc+b*L+j);')
        s=s.replace('const float scale_log2 = p.scale_log2 * (*p.k_scale);','const float scale_log2 = p.scale_log2 * ksc;')
        s=s.replace('const float vs = *p.v_scale;','const float vs = vsc;')
        return CUDA_HELPERS+s
    if name=='VQ8_CUDA':
        s=once(s,'bool virt, float ksc, float vsc) {','bool virt, float ksc, float vsc, float invf, bool first, int64_t first_loc0, int64_t first_loc1) {')
        s=once(s,'  const float invf = p.window ? p.inv_freq[lane] : 0.f;\n','')
        s=once(s,'const int64_t row0 = p.loc[tk[16]], row1 = p.loc[tk[17]];',
            'const int64_t row0 = first ? first_loc0 : p.loc[tk[16]], row1 = first ? first_loc1 : p.loc[tk[17]];')
        s=once(s,'const float ksc = *p.ks, vsc = *p.vs, sc = p.sc * ksc;',
            'const float ksc = pa_f32(p.ks), vsc = pa_f32(p.vs), sc = p.sc * ksc;\n  const float invf = p.window ? pa_f32(p.inv_freq+lane) : 0.f;')
        s=once(s,'  const bool live = 2 * w < p.L;',
            '  const bool live = 2 * w < p.L;\n  const int first_b = S.po_s[0]/p.HP;\n  const int64_t first_loc0 = pa_i64(p.loc+first_b*p.L+min(2*w,p.L-1));\n  const int64_t first_loc1 = pa_i64(p.loc+first_b*p.L+min(2*w+1,p.L-1));')
        s=once(s,'virt, ksc, vsc);','virt, ksc, vsc, invf, po==0, first_loc0, first_loc1);')
        a=s.index('__global__ void __launch_bounds__(256, 4) vq8_combine')
        pre,post=s[:a],s[a:]
        post=once(post,'  const float vsc = *p.vs;\n','')
        post=once(post,'  asm volatile("griddepcontrol.wait;"','  const float vsc = pa_f32(p.vs);\n  asm volatile("griddepcontrol.wait;"')
        return CUDA_HELPERS+pre+post
    if name=='SMALLM_VAR_CUDA':
        s=once(s,'  if constexpr (PDL) asm volatile("griddepcontrol.wait;"',
             '  const float pa_gs13 = pa_f32(gs13+lane), pa_gs2 = pa_f32(gs2+lane);\n  if constexpr (PDL) asm volatile("griddepcontrol.wait;"')
        s=s.replace('gs13[e] * 256.f','__shfl_sync(0xffffffffu,pa_gs13,e) * 256.f')
        s=s.replace('gs2[e] * 256.f','__shfl_sync(0xffffffffu,pa_gs2,e) * 256.f')
        # The stripped native source defines int64_t through its first include.
        anchor='namespace {'
        return once(s,anchor,CUDA_HELPERS+'\n'+anchor)
    return s

def triton_source(name,s):
    if name in ('_boundary_fast','_route_fast','_best7_old_boundary_kernel','_best7_old_route_norm_kernel','_qk_prep_kernel'):
        # Only the listed immutable/timed-input pointers; BR/PR/STREAM/QKV stay dependent.
        for ptr in ('ALPHA','BETA','PREW','POSTW','BIAS','KSCALE','VSCALE','INV_FREQ'):
            s=s.replace('tl.load('+ptr, '_pa_f32('+ptr)
        s=s.replace('tl.load(LSTART)', '_pa_i32(LSTART)').replace('tl.load(POS+t)','_pa_i64(POS+t)').replace('tl.load(LOC+t)','_pa_i64(LOC+t)')
    if name=='_boundary_late':
        s=once(s,'    _pdl_wait(USE_PDL)\n    j = tl.arange(0, 8192)',
            '    j = tl.arange(0, 8192)\n    alpha=_pa_f32(ALPHA+j);beta=_pa_f32(BETA+j);prew=_pa_f32(PREW+j)\n    if HAS_POSTW:postw=_pa_f32(POSTW+j)\n    _pdl_wait(USE_PDL)')
        for ptr,var in (('ALPHA','alpha'),('BETA','beta'),('PREW','prew'),('POSTW','postw')):s=s.replace('tl.load('+ptr+' + j)',var)
    if name=='_best7_old_qk_prep_kernel':
        pre='    cache_row=_pa_i64(LOC+t);kscale=_pa_f32(KSCALE);vscale=_pa_f32(VSCALE)\n    if ROPE:\n        half=tl.arange(0,D//2)\n        angle=_pa_i64(POS+t).to(tl.float32)*_pa_f32(INV_FREQ+half)\n        rope_cos=tl.cos(angle)[None,:];rope_sin=tl.sin(angle)[None,:]\n'
        s=once(s,'    _pdl_wait(USE_PDL)',pre+'    _pdl_wait(USE_PDL)')
        a=s.index('        pos = tl.load(POS');b=s.index('        qe, qd =',a)
        s=s[:a]+'        c=rope_cos;sn=rope_sin\n'+s[b:]
        s=s.replace('row = tl.load(LOC + t)','row = cache_row').replace('tl.load(KSCALE)','kscale').replace('tl.load(VSCALE)','vscale')
    if name=='_attn_combine_kernel':
        s=once(s,'    _pdl_wait(USE_PDL)','    vscale=_pa_f32(VSCALE)\n    _pdl_wait(USE_PDL)')
        s=s.replace('* tl.load(VSCALE)','* vscale')
    if name=='_attn_split_kernel':
        s=s.replace('seq = tl.load(SEQ + b)','seq = _pa_i32(SEQ + b)')
        pre='''    pa_kscale=_pa_f32(KSCALE)
    pa_pos=start+n
    pa_valid=pa_pos<end
    if PAGE == 1:
        pa_rows=_pa_keepi(tl.load(PT+b*pt_stride+pa_pos,pa_valid,0))
    else:
        pa_pages=_pa_keepi(tl.load(PT+b*pt_stride+pa_pos//PAGE,pa_valid,0))
        pa_rows=pa_pages*PAGE+pa_pos%PAGE
    pa_rows=pa_rows.to(tl.int64)
    pa_hist=pa_valid & (pa_pos<seq)
    pa_kt=_pa_keep(tl.load(KC+pa_rows[:,None]*W2+c[None,:],pa_hist[:,None],0.).to(tl.float32)).to(tl.bfloat16)
    pa_vt=_pa_keep(tl.load(VC+pa_rows[:,None]*W2+c[None,:],pa_hist[:,None],0.).to(tl.float32)).to(tl.bfloat16)
'''
        s=once(s,'    _pdl_wait(USE_PDL)',pre+'    _pdl_wait(USE_PDL)')
        s=s.replace('sm_scale * tl.load(KSCALE)','sm_scale * pa_kscale')
        a=s.index('        if PAGE == 1:');b=s.index('        sc = tl.dot',a)
        original=s[a:b]
        first='''        if kb == start:
            prow=pa_rows
            pa_new=valid & (kpos>=seq)
            new_kt=tl.load(KC+prow[:,None]*W2+c[None,:],pa_new[:,None],0.).to(tl.bfloat16)
            new_vt=tl.load(VC+prow[:,None]*W2+c[None,:],pa_new[:,None],0.).to(tl.bfloat16)
            kt=tl.where((kpos<seq)[:,None],pa_kt,new_kt)
            vt=tl.where((kpos<seq)[:,None],pa_vt,new_vt)
        else:
'''
        s=s[:a]+first+''.join('    '+line+'\n' for line in original.splitlines())+s[b:]
    return s

def patch(src):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'requires S14 final f1c77ce6'
    lines=src.splitlines(True);changes=[]
    native_names=('GLUE_CL_CUDA','CUDA_SRC','PARTIAL_CUDA','VQ8_CUDA','SMALLM_VAR_CUDA')
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in native_names:
            name=n.targets[0].id;old=ast.literal_eval(n.value);new=native(name,old)
            changes.append((n.lineno-1,n.end_lineno,name+' = '+repr(new)+'\n'))
        elif isinstance(n,ast.FunctionDef):
            old=ast.get_source_segment(src,n);new=triton_source(n.name,old)
            if old!=new:changes.append((n.lineno-1,n.end_lineno,new+'\n'))
    for lo,hi,new in sorted(changes,reverse=True):lines[lo:hi]=[new]
    # Definitions precede every JIT kernel; no duplicate source/module or runtime dispatch.
    out=''.join(lines);anchor='@triton.jit\ndef _l2_prefetch'
    out=once(out,anchor,TRITON_HELPERS+'\n'+anchor)
    ast.parse(out)
    return out
