"""Tune the H_H expert threshold and heavy tile independently for each phase."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'phase_mix';ROOT=HERE.parents[5]
sys.path[:0]=[str(HERE.parents[1]),str(ROOT/'tools')]
from build_prefill_lab import subset
from native_abi import CPP_PRELUDE
from okc_run import rejected_words


def sources():
    admitted=json.loads((BASE/'hh_handoff.json').read_text())
    for p,sha in admitted['embedded_sources'].items():assert hashlib.sha256((BASE/(p+'.cu')).read_bytes()).hexdigest()==sha
    dist=(BASE/'idist.cu').read_text().replace('idist','swdist')
    old='F(0,16,128,4,1) F(1,32,128,4,1)';assert old in dist
    dist=dist.replace(old,'F(1,32,128,4,1) F(2,64,128,4,1)')
    nt=(BASE/'pm_nt.cu').read_text().replace('pmnt','swnt').replace('pm_nt','sw_nt')
    assert 'if(n>16)nb=0;' in nt
    nt=nt.replace('if(n>16)nb=0;','if(n>threshold)nb=0;')
    nt=nt.replace('float*y,float*ss,int M,int nstage){','float*y,float*ss,int M,int nstage,int threshold){')
    old='int64_t stages,int64_t grid,int64_t stream){';assert nt.count(old)==1
    nt=nt.replace(old,'int64_t stages,int64_t grid,int64_t threshold,int64_t stream){')
    old='(float*)v[11],M,stages);';assert nt.count(old)==1
    nt=nt.replace(old,'(float*)v[11],M,stages,threshold);')
    meta='''
#include <cuda_runtime.h>
#include <vector>
__global__ void sw_meta_kernel(const int*C,const int*ORDER,int*T13,int*T2,float*SS,int cap,int h13,int n13,int h2,int n2){
 int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;
 for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],a=n>h13?(n+n13-1)/n13:0,b=n>h2?(n+n2-1)/n2:0;
  for(int d=1;d<32;d*=2){int aa=__shfl_up_sync(~0u,a,d),bb=__shfl_up_sync(~0u,b,d);if(t>=d){a+=aa;b+=bb;}}
  T13[t+1]=a;T2[t+1]=b;if(t==0){T13[0]=0;T2[0]=0;}
 }
}
void sw_meta(std::vector<int64_t>v,int64_t cap,int64_t h13,int64_t n13,int64_t h2,int64_t n2,int64_t stream){
 sw_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(float*)v[4],cap,h13,n13,h2,n2);}
'''
    return dict(swdist=dist,sw_nt=nt,sw_meta=meta)


def main():
    codes=sources();cpp=CPP_PRELUDE;funs=[]
    cpp+='int64_t swdist_init();std::vector<int64_t> swdist_info(int64_t,int64_t,int64_t);void swdist_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void swdist_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void swdist_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\n'
    funs+=['swdist_'+f for f in ('init','info','route','run','finish')]
    cpp+='int64_t sw_nt_init();int64_t sw_nt_smem(int64_t,int64_t);int64_t sw_nt_resident(int64_t,int64_t,int64_t,int64_t);void sw_nt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void sw_meta(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
    funs+=['sw_nt_'+f for f in ('init','smem','resident','run')]+['sw_meta']
    control=(BASE/'hh_cell.py').read_text();assert hashlib.sha256(control.encode()).hexdigest()=='970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f'
    helpers=subset((ROOT/'prof/lab/moe_m128_phase_mix.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph','pm_error'))
    # Independent fixtures/oracles keep their original names; the admitted cell
    # has namespaced definitions and no experimental branch selection.
    text=helpers+'\n'+control+'\nimport statistics\nimport time\nimport gc\n'
    cuda='\n'.join(codes.values())
    text+='SW_CUDA='+repr(cuda)+'\nSW_CPP='+repr(cpp)+'\nSW_FUNCTIONS='+repr(funs)+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    for p,c in codes.items():(HERE/(p+'.cu')).write_text(c)
    out=ROOT/'prof/lab/moe_m128_mix_sweep.py';out.write_text(text)
    meta=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),sources={p:hashlib.sha256(c.encode()).hexdigest() for p,c in codes.items()},
        control_cell_sha256=hashlib.sha256(control.encode()).hexdigest(),H200_control='770809cf H_H234.44us',thresholds=[8,16,24,32],heavy_tiles=[32,64],
        phase_choices='W13-only sweep, W2-only sweep, coupled sweep; all-NT1/all-WGMMA W2 controls',host='muse_bf16_proj',not_before='2026-10-25T09:01:00+09:00')
    (HERE/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['payload'],meta['sha256'],len(text))


if __name__=='__main__':main()
