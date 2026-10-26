"""S18 activation reuse with the admitted CHUNK4 token tiling and P7 shell."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma,zlib
from pathlib import Path
H=Path(__file__).resolve().parent;B=H.parent;ROOT=H.parents[5]
spec=importlib.util.spec_from_file_location('act_parent',B/'s15_review/build.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
value,function,once=parent.value,parent.function,parent.once
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best7_s18_final.py'
ARMS=('S18','PACK','TMA1','MC2','MC4','W4N32','W4N64')
def source():
 s=BASE.read_text();n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode')
 s=lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode();assert hashlib.sha256(s.encode()).hexdigest()=='95df42fe7b3a31e66d60bc769bae15412c952ad77753b7d15b3beea3d89b3af4';return s

def tiles(base):
 cu=value(base,'P3B_CUDA');a=cu.index('template<int NT,bool F,int CS,int S>');e=cu.index('\n\n\ntemplate<int DEPTH,bool F>',a);raw=cu[a:e]
 pack=raw.replace('void tile(','void packed_tile(')
 start=pack.index(' if constexpr(F){\n  [&]');end=pack.index('  if(t<64){',start)
 pack=pack[:start]+''' const __nv_bfloat16*copy_root=X+(size_t)(offset+row+t/8)*4096+(t&7)*8;
 auto issue=[&](int k,int st){
  if(wg==0)[&]<int...J>(std::integer_sequence<int,J...>){
   (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);
   (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);
  }(std::make_integer_sequence<int,NJ>{});copy_root+=128;
'''+pack[end:]
 mc=pack.replace('void packed_tile(','void multicast_tile(')
 start=mc.index(' const __nv_bfloat16*copy_root=');end=mc.index('  if(t<64){',start)
 mc=mc[:start]+''' const __nv_bfloat16*copy_root=X+(size_t)(offset+row+t/8)*4096+(t&7)*8;
#if BR_HOPPER
 if(threadIdx.x==0){for(int j=0;j<S;++j)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(smaddr(bar+j)):"memory");asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 __syncthreads();chunk::cb();
#endif
 auto issue=[&](int k,int st){
#if BR_HOPPER
  if(threadIdx.x==0)asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(smaddr(bar+st)),"r"(NT*KT*2):"memory");
  chunk::cb();
  if(threadIdx.x==0&&rank==0){
   #pragma unroll
   for(int half=0;half<2;++half)asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0],[%1,{%2,%3}],[%4],%5;"::"r"(smaddr(bs+st*NT*KT+half*NT*64)),"l"(map),"r"(k*KT+half*64),"r"(offset+row),"r"(smaddr(bar+st)),"h"((unsigned short)((1u<<CS)-1)):"memory");
  }
#else
  if(wg==0)[&]<int...J>(std::integer_sequence<int,J...>){
   (copyx<NT,J,0>(bs,copy_root,st,count-row-t/8),...);
   (copyx<NT,J,1>(bs,copy_root,st,count-row-t/8),...);
  }(std::make_integer_sequence<int,NJ>{});copy_root+=128;
#endif
'''+mc[end:]
 mc=once(mc,'  __syncthreads();\n#if BR_HOPPER\n  asm volatile("fence.proxy','  __syncthreads();\n#if BR_HOPPER\n  chunk::mbwait(smaddr(bar+st),(k/S)&1);\n  asm volatile("fence.proxy')
 mc=mc.rsplit(' __syncthreads();',1)[0]+''' __syncthreads();
#if BR_HOPPER
 chunk::cb();
 if(threadIdx.x==0)for(int j=0;j<S;++j)asm volatile("mbarrier.inval.shared::cta.b64 [%0];"::"r"(smaddr(bar+j)):"memory");
 chunk::cb();
#endif
}
'''
 wide=raw.replace('void tile(','void wide_tile(')
 wide=once(wide,'BS=(S*WG*(QB+SB)+1023)&~1023','BS=(S*WG*2*(QB+SB)+1023)&~1023')
 wide=once(wide,'auto*ss=qs+S*WG*QB','auto*ss=qs+S*WG*2*QB')
 wide=once(wide,' const int*qg=Q;const unsigned char*sg=SC;',' const int*qg[2];const unsigned char*sg[2];\n #pragma unroll\n for(int o=0;o<2;++o){')
 wide=once(wide,'if(t<64)qg=Q+','if(t<64)qg[o]=Q+').replace('mt*4+(t>>4)','(mt+o*2)*4+(t>>4)')
 wide=once(wide,'sg=SC+','sg[o]=SC+').replace('mt*4+i/8','(mt+o*2)*4+i/8')
 wide=once(wide,' unsigned qd=smaddr(',' }\n unsigned qd=smaddr(')
 start=wide.index('  if(t<64){',wide.index(' auto issue='));end=wide.index('  asm volatile("cp.async.commit_group;',start)
 issue=wide[start:end].replace('qd+st*WG*QB','qd+(st*2+o)*WG*QB').replace('qg+g*64','qg[o]+g*64').replace('qg+=256','qg[o]+=256').replace('sd+st*WG*SB','sd+(st*2+o)*WG*SB').replace('"l"(sg)','"l"(sg[o])').replace('sg+=128','sg[o]+=128')
 wide=wide[:start]+'  #pragma unroll\n  for(int o=0;o<2;++o){\n'+issue+'  }\n'+wide[end:]
 wide=once(wide,'float acc[NT/2]={};','float accum[2][NT/2]={};')
 start=wide.index('  uint4 sv=');end=wide.index('#if BR_HOPPER\n  if constexpr(S==1)',start)
 math=wide[start:end].replace('(st*WG+wg)','((st*2+o)*WG+wg)').replace('if(sub==0){','if(o==1&&sub==0){')
 wide=wide[:start]+'  #pragma unroll\n  for(int o=0;o<2;++o){float*acc=accum[o];\n'+math+'  }\n'+wide[end:]
 start=wide.index(' if constexpr(F){float gg=');end=wide.rindex(' __syncthreads();')
 epi=wide[start:end].replace('int ch=mt*32','int ch=(mt+o*2)*32').replace('cc=mt*64','cc=(mt+o*2)*64').replace('*64+mt]','*64+mt+o*2]')
 wide=wide[:start]+' #pragma unroll\n for(int o=0;o<2;++o){float*acc=accum[o];\n'+epi+' }\n'+wide[end:]
 return cu,'\nnamespace ax {using namespace chunk;\n'+pack+'\n'+mc+'\n'+wide+'\n'+(H/'kernels.cu.inc').read_text()

def main():
 base=source();cu,extra=tiles(base);cu+='#include <cudaTypedefs.h>\n'+extra+'\n'+(H/'host.cu.inc').read_text();(H/'activation.cu').write_text(cu)
 cpp=value(base,'P3B_CPP')+'''void ax_gather(std::vector<int64_t>,int64_t,int64_t,int64_t);void ax_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> ax_info(int64_t,int64_t);'''
 fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','ch_info','ch_run','ax_gather','ax_run','ax_info']
 plan=function(base,'p3b_plan').replace('def p3b_plan(data,base,native,streams):','def activation_plan(data,base,native,streams,arm):').replace('mod=p3b_module()','mod=activation_module()')
 plan=once(plan,'    p13,p2,pn=',"    original_h13=list(h13);gx=torch.empty_like(act) if arm in ('PACK','TMA1','MC2','MC4') else act\n    if arm in ('PACK','TMA1','MC2','MC4'):h13[3]=gx\n    p13,p2,pn=")
 plan=once(plan,'owners=(h13,h2,nv,ids,tw,pos,tasks,l13,l2)','owners=(h13,h2,nv,ids,tw,pos,tasks,l13,l2,gx,original_h13)')
 plan=once(plan,'l13=list(h13);l2=list(h2);','l13=list(h13);l13[3]=x;l2=list(h2);')
 plan=once(plan,'    def route():','    gather_args=[t.data_ptr() for t in original_h13]\n    kind={"PACK":0,"TMA1":1,"MC2":2,"MC4":4,"W4N32":5,"W4N64":6}[arm]\n    def route():')
 plan=once(plan,'        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)','        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)\n        if arm in ("PACK","TMA1","MC2","MC4"):mod.ax_gather(gather_args,gx.data_ptr(),m,torch.cuda.current_stream().cuda_stream)')
 plan=once(plan,'mod.ch_run(p13,1,4,sms*4,m,side.cuda_stream);mod.ch_run(p2,0,4,sms*4,m,side.cuda_stream)','mod.ax_run(p13,kind,1,m,side.cuda_stream);mod.ax_run(p2,kind,0,m,side.cuda_stream)')
 cell='ACT_CUDA='+repr(cu)+'\nACT_CPP='+repr(cpp)+'\nACT_FUNCTIONS='+repr(fs)+'\n_ACT_MODULE=[]\n'
 cell+='''def activation_module():
 if not _ACT_MODULE:
  arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  mod=load_inline(name='p3b_act_'+hashlib.sha256(ACT_CUDA.encode()).hexdigest()[:12],cpp_sources=ACT_CPP,cuda_sources=ACT_CUDA,functions=ACT_FUNCTIONS,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
  mod.br_init();NF_OPTS['pmz']=mod.pmz_init()
  for kind in (0,1,2,4,5,6):
   for first in (0,1):mod.ax_info(kind,first)
  _ACT_MODULE.append(mod)
 return _ACT_MODULE[0]
'''+plan+'\n'
 (H/'activation_cell.py').write_text(cell);out=ROOT/'runs/screen/p3b_actreuse';out.mkdir(parents=True,exist_ok=True);manifest={}
 for arm in ARMS[1:]:
  hook=f'''\n_activation_old_nf_call=nf_call
def nf_call(data,state):
 for fn in activation_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'],{arm!r})['steps']:out=fn()
 return out
_activation_old_s4=_s4_prepare_banks
def _s4_prepare_banks(state,weights):
 _activation_old_s4(state,weights);activation_module();return state
'''
  candidate=base+'\n'+cell+hook;p=out/(arm.lower()+'.py');p.write_text(candidate)
  digest=hashlib.sha256(candidate.encode()).hexdigest();encoded=base64.b85encode(lzma.compress(candidate.encode(),preset=9)).decode()
  packed='# okc: problem=muse_decode_layer\nimport torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(digest)+'\n_file='+repr('<p3b-act-'+arm+'>')+'\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"))\n'
  packed_path=out/(arm.lower()+'_packed.py');packed_path.write_text(packed);assert len(packed)<1000000
  patch=''.join(difflib.unified_diff(BASE.read_text().splitlines(True),packed.splitlines(True),fromfile='a/'+str(BASE.relative_to(ROOT)),tofile='b/'+str(BASE.relative_to(ROOT))))
  (H/(arm.lower()+'.patch')).write_text(patch);manifest[arm]=dict(candidate=str(p.relative_to(ROOT)),candidate_sha256=digest,packed_candidate=str(packed_path.relative_to(ROOT)),packed_sha256=hashlib.sha256(packed.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
 manifest.update(base_expanded_sha256=hashlib.sha256(base.encode()).hexdigest(),base_shipped_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),arms=ARMS)
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
if __name__=='__main__':main()
