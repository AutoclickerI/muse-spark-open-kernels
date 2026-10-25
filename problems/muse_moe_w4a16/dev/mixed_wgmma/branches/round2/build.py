"""Build one round-two packet: two transposed tiles and intra-CTA X reuse."""
import ast
import hashlib
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;B=H.parent;R=B.parents[4]
sys.path.insert(0,str(B))
import cluster_build
import cluster_copy_build
from cluster_schedule_build import deep_cuda
from light_build import once,value


def cta2(cu):
 a=cu.index('namespace oc {');z=cu.index('// Host entry points',a) if '// Host entry points' in cu[a:] else cu.index('static CUtensorMap oc_map(',a)
 dev=cu[a:z];host=cu[z:]
 dev=once(dev,'constexpr int KT=128,QB=4096,SB=512;','constexpr int KT=128,QB=4096,SB=512,WG=2;')
 dev=dev.replace('S*(QB+SB)','S*WG*(QB+SB)').replace('S*QB','S*WG*QB')
 dev=once(dev,'int t=threadIdx.x,w=t>>5,l=t&31,count=C[e]', 'int t=threadIdx.x&127,w=t>>5,l=t&31,wg=threadIdx.x>>7,count=C[e]')
 dev=once(dev,'auto*qs=ocsm;auto*ss=qs+S*WG*QB;', 'auto*qs=ocsm;auto*ss=qs+S*WG*QB;')
 dev=once(dev,'unsigned qd=smaddr(qs+(t>>4)*512+(t&15)*16),sd=smaddr(ss+(t-64)*16);', 'unsigned qd=smaddr(qs+wg*QB+(t>>4)*512+(t&15)*16),sd=smaddr(ss+wg*SB+(t-64)*16);')
 dev=dev.replace('qd+st*QB','qd+st*WG*QB').replace('sd+st*SB','sd+st*WG*SB')
 dev=dev.replace('ss+st*SB+(w*8+(l>>2))*16','ss+(st*WG+wg)*SB+(w*8+(l>>2))*16')
 dev=dev.replace('qs+st*QB+sub*2048+t*16','qs+(st*WG+wg)*QB+sub*2048+t*16')
 dev=once(dev,'  [&]<int...J>(std::integer_sequence<int,J...>){\n   if constexpr(F)', '  if(wg==0)[&]<int...J>(std::integer_sequence<int,J...>){\n   if constexpr(F)')
 dev=once(dev,'void __launch_bounds__(128,2) kernel(', 'void __launch_bounds__(256,1) kernel(')
 dev=once(dev,'MT=F?128:64','MT=F?64:32')
 dev=once(dev,'float*red=(float*)ocsm;', 'float*red=(float*)ocsm+(threadIdx.x>>7)*4*MAX;')
 dev=once(dev,'e,row,mt,bar,','e,row,mt*WG+(threadIdx.x>>7),bar,')
 a=host.index(' if(major==9){BR_CUDA_CHECK(cudaLaunchKernelEx');z=host.index('\n}',a)
 host=host[:a]+' '+host[host.index('else{',a)+5:z-1]+host[z:]
 host=host.replace(' F(64,4) F(128,4) F(192,4) F(256,4)','')
 host=host.replace('cfg.blockDim=dim3(128)','cfg.blockDim=dim3(256)').replace('<<<grid,128,oc::dyn','<<<grid,256,oc::dyn').replace('>,128,oc::dyn','>,256,oc::dyn')
 return cu[:cu.index('namespace oc {')]+dev+host


def main():
 # Rebuild generated dependencies; the already queued reset wrapper is immutable.
 cluster_copy_build.main()
 source=(B/'cluster_payload.py').read_text();tree=ast.parse(source)
 prefix=source[:source.index('OP_CUDA=')]
 # No cluster runtime is needed: only its proven fixture/bank helpers and control plan.
 tree=ast.parse(prefix);tree.body=[n for n in tree.body if not(isinstance(n,ast.FunctionDef) and n.name in ('oc_plan','nf_plan','nf_call'))];prefix=ast.unparse(tree)+'\n'
 base=value(source,'OP_CUDA')[1];assert hashlib.sha256(base.encode()).hexdigest().startswith('0a31e251')
 s14=(R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text();assert hashlib.sha256(s14.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac';assert value(s14,'NF_CUDA')==base
 trans=base+'\n'+(H/'transpose.cu.inc').read_text().replace('RT_MMA',cluster_build.mma())
 direct=(B/'cu_cluster_direct.cu').read_text();dual=cta2(direct);deep=deep_cuda(dual)
 cpp=value(s14,'NF_CPP');tcpp='void rt_pack(std::vector<int64_t>,int64_t,int64_t,int64_t);void rt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> rt_info(int64_t,int64_t,int64_t);'
 body=prefix+'\nOP_CUDA='+repr([base,trans,trans,dual,deep])+'\nOP_CPP='+repr([cpp,cpp+tcpp,cpp+tcpp,cpp+cluster_build.CPP,cpp+cluster_build.CPP])+'\nNF_OPTS={}\n'+(H/'runtime.py.inc').read_text()
 compile(body,'<round2>','exec');(H/'payload.py').write_text(body)
 for arm in (1,2,3,4):
  screen='# okc: problem=muse_moe_w4a16\ndef lab_cold_l2():pass\n'+body+'\nSCREEN_ARM='+str(arm)+'\n'+'''def prepare(weights):
    op_build()
    w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in weights.items()}
    w['post_expert_norm']=w['post_norm_w']
    return op_prepare(w)
def kernel_fn(data,state):return op_call(data,state,SCREEN_ARM)
'''
  (H/('screen_'+('trans64','trans128','cta2','cta2s3')[arm-1]+'.py')).write_text(screen)
 (H/'transpose.cu').write_text(trans);(H/'cta2.cu').write_text(dual);(H/'cta2s3.cu').write_text(deep)
 manifest=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),base_sha256=hashlib.sha256(s14.encode()).hexdigest(),transpose_cuda_sha256=hashlib.sha256(trans.encode()).hexdigest(),cta2_cuda_sha256=hashlib.sha256(dual.encode()).hexdigest(),cta2s3_cuda_sha256=hashlib.sha256(deep.encode()).hexdigest())
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)

if __name__=='__main__':main()
