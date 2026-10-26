"""S19: constant legal FA3 schedules, D exponent scheduling, and C3 page lookahead."""
import difflib,hashlib,json,re
from pathlib import Path
from build_fixed import compact
H=Path(__file__).resolve().parent;o=H/'phases';o.mkdir(exist_ok=True)
b=json.loads((o/'base_bundle.json').read_text())
def one(c,a,z):
 assert c.count(a)==1,(a[:80],c.count(a));return c.replace(a,z)
def mixed(c,fraction):
 helper=r'''
// Cody-Waite nearest integer range reduction; pure registers, no artificial
// memory clobbers around the polynomial or probability pair. FP32 throughout.
DI float phase_poly(float x){
 float xc=fmaxf(x,-126.f),r=xc+12582912.f,n=r-12582912.f,f=xc-n;
 float p=fmaf(f,0.05587551f,0.24223916f);
 p=fmaf(p,f,0.69314718f);p=fmaf(p,f,0.99994826f);
 return __uint_as_float(__float_as_uint(p)+(__float_as_uint(r)<<23));
}
'''
 c=one(c,'// Online softmax of one 64-key tile;',helper+'\n// Online softmax of one 64-key tile;')
 a='const float p0 = ex2(fmaf(s[n][2 * rh], sc, bias)), p1 = ex2(fmaf(s[n][2 * rh + 1], sc, bias));'
 p='phase_poly(fmaf(s[n][2 * rh],sc,bias))'
 if fraction==25:p='((n&1)==0?'+p+':ex2(fmaf(s[n][2 * rh],sc,bias)))'
 return one(c,a,'const float p0 = '+p+', p1 = ex2(fmaf(s[n][2 * rh + 1],sc,bias));')
c3=compact(b['ATTN_PORT_CUDA_SRC'],3)
def transport(c,depth,vector):
 c=one(c,'constexpr int PD=8;',f'constexpr int PD={depth};')
 if vector:
  a=c.index('    if(threadIdx.x<64){',c.index('auto page_issue='));z=c.index('\n  };',a)
  c=c[:a]+'''    const int key0=c_lo+tile*T;
    // Four page IDs per instruction when the whole quad is in bounds. The
    // scalar tail handles arbitrary nearby lengths, offsets and page sizes.
    if(p.ps==1 && (reinterpret_cast<uintptr_t>(pt+key0)&15)==0 && key0+63<seq){
      if(threadIdx.x<16){const int key=key0+4*threadIdx.x;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(smem_u32(&page_ring[tile%PD][4*threadIdx.x])),"l"(pt+key):"memory");}
    }else if(threadIdx.x<64){
      const int key=key0+threadIdx.x,clamped=min(key,max(seq-1,0));
      asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"::"r"(smem_u32(&page_ring[tile%PD][threadIdx.x])),"l"(pt+(clamped>>p.ps_log2)):"memory");
    }'''+c[z:]
 c=one(c,'__launch_bounds__(NTHR, 2) attn_main', '__launch_bounds__(NTHR, L==1?4:2) attn_main')
 return c
kernels={'Aq':b['ATTN_PORT_CUDA_SRC'],'Av':b['VQ8_CUDA'],'P':b['PARTIAL_CUDA'],'C3':c3,
 'T8':transport(c3,8,True),'T16':transport(c3,16,True),'E25':mixed(b['VQ8_CUDA'],25),'E50':mixed(b['VQ8_CUDA'],50)}
meta={}
for name,c in kernels.items():
 (o/(name+'.cu')).write_text(c);meta[name]=dict(sha256=hashlib.sha256(c.encode()).hexdigest())
 if name in ('C3','T8','T16','E25','E50'):
  old=b['VQ8_CUDA'] if name.startswith('E') else b['ATTN_PORT_CUDA_SRC']
  patch=''.join(difflib.unified_diff(old.splitlines(True),c.splitlines(True),fromfile='a/'+('VQ8_CUDA' if name.startswith('E') else 'ATTN_PORT_CUDA_SRC'),tofile='b/'+name))
  (o/(name+'.cuda.patch')).write_text(patch)
(o/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
