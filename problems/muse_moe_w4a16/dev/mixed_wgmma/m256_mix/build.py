"""M256 expert/phase composition on packed uint8 scale banks only."""
import ast
import hashlib
import json
import sys
from pathlib import Path

H=Path(__file__).resolve().parent;P=H.parent;R=H.parents[4]
sys.path[:0]=[str(P),str(R/'tools')]
from build_prefill_lab import subset
from native_abi import CPP_PRELUDE
from okc_run import rejected_words


def sources():
    c1=(P/'recode/recode.cu').read_text()
    assert hashlib.sha256(c1.encode()).hexdigest()=='c16b1aa0161301ad281d6e4cb50a288c2458715c5a6ba30592ed3ed2b40bbda0'
    old='F(0,16,128,4,1) F(1,32,128,4,1)'
    assert old in c1;c1=c1.replace(old,'F(1,32,128,4,1) F(2,64,128,4,1)')
    nt=(P/'m128/phase_mix/pm_nt.cu').read_text().replace('pmnt','mxnt').replace('pm_nt','mx_nt')
    assert 'if(n>16)nb=0;' in nt
    nt=nt.replace('if(n>16)nb=0;','if(n>threshold)nb=0;')
    nt=nt.replace('float*y,float*ss,int M,int nstage){','float*y,float*ss,int M,int nstage,int threshold){')
    nt=nt.replace('int64_t stages,int64_t grid,int64_t stream){','int64_t stages,int64_t grid,int64_t threshold,int64_t stream){')
    nt=nt.replace('(float*)v[11],M,stages);','(float*)v[11],M,stages,threshold);')
    opt=(P/'w13_issue/issue.cu').read_text();opt=opt[:opt.index('int64_t wi_init')].replace('mwwi','mxopt')
    # The W13 receipt is not available yet. Keep both vector and quad bodies
    # locally checked; the final payload's fixed decoder is selected on receipt.
    opt+='''
int64_t mx_opt_init(){
 auto set=[&]<int N,bool Q>(){constexpr int bytes=1023+18432+4*N*128*2;WX_CUDA_CHECK(cudaFuncSetAttribute(mxopt::fused<N,128,4,1,true,1,true,true,0,Q>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};
 set.template operator()<32,false>();set.template operator()<64,false>();set.template operator()<32,true>();set.template operator()<64,true>();return 0;}
std::vector<int64_t> mx_opt_info(int64_t n,int64_t mode){
 auto info=[&]<int N,bool Q>(){constexpr int bytes=1023+18432+4*N*128*2;cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mxopt::fused<N,128,4,1,true,1,true,true,0,Q>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mxopt::fused<N,128,4,1,true,1,true,true,0,Q>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};
 if(n==32)return mode==1?info.template operator()<32,false>():info.template operator()<32,true>();WX_CHECK(n==64);return mode==1?info.template operator()<64,false>():info.template operator()<64,true>();}
void mx_opt_run(std::vector<int64_t>v,int64_t n,int64_t mode,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){
 auto run=[&]<int N,bool Q>(){constexpr int bytes=1023+18432+4*N*128*2;mxopt::fused<N,128,4,1,true,1,true,true,0,Q><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};
 if(n==32){if(mode==1)run.template operator()<32,false>();else run.template operator()<32,true>();}else{WX_CHECK(n==64);if(mode==1)run.template operator()<64,false>();else run.template operator()<64,true>();}WX_CUDA_CHECK(cudaGetLastError());}
'''
    meta='''
#include <cuda_runtime.h>
#include <vector>
__global__ void mx_meta_kernel(const int*C,const int*ORDER,int*T13,int*T2,float*SS,int cap,int h13,int n13,int h2,int n2){
 int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;
 for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],a=n>h13?(n+n13-1)/n13:0,b=n>h2?(n+n2-1)/n2:0;
  for(int d=1;d<32;d*=2){int aa=__shfl_up_sync(~0u,a,d),bb=__shfl_up_sync(~0u,b,d);if(t>=d){a+=aa;b+=bb;}}
  T13[t+1]=a;T2[t+1]=b;if(t==0){T13[0]=0;T2[0]=0;}
 }
}
void mx_meta(std::vector<int64_t>v,int64_t cap,int64_t h13,int64_t n13,int64_t h2,int64_t n2,int64_t stream){mx_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(float*)v[4],cap,h13,n13,h2,n2);}
'''
    return dict(c1=c1,nt=nt,opt=opt,meta=meta)


def main():
    codes=sources();cpp=CPP_PRELUDE
    cpp+='''
int64_t rc_init();std::vector<int64_t> rc_info(int64_t,int64_t,int64_t);void rc_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void rc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void rc_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);
int64_t mx_nt_init();int64_t mx_nt_smem(int64_t,int64_t);int64_t mx_nt_resident(int64_t,int64_t,int64_t,int64_t);void mx_nt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
int64_t mx_opt_init();std::vector<int64_t> mx_opt_info(int64_t,int64_t);void mx_opt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
void mx_meta(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);
'''
    functions=['rc_'+f for f in ('init','info','route','run','finish')]+['mx_nt_'+f for f in ('init','smem','resident','run')]+['mx_opt_'+f for f in ('init','info','run')]+['mx_meta']
    fixture=subset((R/'prof/lab/moe_w2fix_m256.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph'))
    packing=subset((P/'engine.py').read_text(),('mw_repack_planes',))
    packing+='\n'+subset((P/'recode/recoded.py').read_text(),('rc_repack','rc_masks','RC_SIGNS','RC_ZEROS'))
    tree=ast.parse(packing);tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ('engine','nt1_pack'))];packing=ast.unparse(ast.fix_missing_locations(tree))
    ntpack=subset((P/'m128/phase_mix/hh_cell.py').read_text(),('m128_hh_repack',))
    # Zero scales need no exponent extrema; preserve exact zero sign/code fold.
    old='assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)'
    assert old in ntpack;ntpack=ntpack.replace(old,'assert not bool(nz.any()) or (int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and int(mant.max()) <= 7)')
    prefix=fixture+'\n'+packing+'\n'+ntpack+'\n'
    prefix+='MX_CUDA='+repr('\n'.join(codes.values()))+'\nMX_CPP='+repr(cpp)+'\nMX_FUNCTIONS='+repr(functions)+'\n'
    adapter=prefix+(H/'adapter.py.inc').read_text();(H/'adapter.py').write_text(adapter)
    body=adapter+'\n'+(H/'driver.py.inc').read_text();assert not rejected_words(body),rejected_words(body)
    out=R/'prof/lab/moe_m256_mix.py';out.write_text(body)
    for name,code in codes.items():(H/(name+'.cu')).write_text(code)
    m=dict(payload=str(out.relative_to(R)),sha256=hashlib.sha256(body.encode()).hexdigest(),sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in codes.items()},
           baseline='9e8a6a15 recoded C1g2112,365.2us',thresholds=[8,16,32],heavy_tiles=[32,64],W13_decoder='pending8483c0c6: baseline0,vector1,quad2 locally screened',not_before='2026-10-25T09:01:00+09:00',status='not sent; awaits W13 receipt and send-time gate')
    (H/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m['payload'],m['sha256'],len(body))


if __name__=='__main__':main()
