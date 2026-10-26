"""One count-routing and one-path batch, with unchanged S15/CH4 controls."""
import ast
import base64
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib
H=Path(__file__).resolve().parent;B=H.parent;ROOT=B.parents[4]
spec=importlib.util.spec_from_file_location('p3b_s15_builder',B/'s15_review/build.py');parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
once,value,function=parent.once,parent.value,parent.function
BASE=parent.BASE
ARMS=('S15','CH4','B8','B4','B2','B0','ONE')

def cuda(base):
 cu=parent.chunk(value(base,'NF_CUDA'))
 cu=once(cu,'F(71,16,false,6,1,8,16)', 'F(71,16,false,6,1,8,16) F(72,16,true,7,1,0,16) F(73,16,false,6,1,0,16)')
 cu=once(cu,'int*T8,int M,int NT,int start){','int*T8,int M,int NT,int start,int threshold=8,int*LOW=nullptr){')
 cu=once(cu,'C[t]=n;', 'C[t]=n;if(LOW)LOW[t]=n<=threshold?n:0;')
 cu=once(cu,'u=ns>8&&ns<=16;', 'u=ns>threshold&&ns<=16;')
 start=cu.index('namespace chunk {');head,chunk=cu[:start],cu[start:]
 chunk=once(chunk,'float*SS,int M){','float*SS,int M,int cutoff){')
 chunk=once(chunk,'if(n<=16)continue;', 'if(n<=cutoff)continue;')
 # Keep the admitted N32/N64 body: a third inlined math case spilled hundreds
 # of bytes on sm90a. Low-count ONE experts deliberately use N32 padding.
 chunk=chunk.replace('(BR_ARGS(v),(int)M)', '(BR_ARGS(v),(int)M,16)')
 cu=head+chunk+'''
void bal_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t low,int64_t threshold,int64_t stream){
 mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start,(int)threshold,(int*)low);
 BR_CUDA_CHECK(cudaGetLastError());}
void bal_run(std::vector<int64_t>v,int64_t first,int64_t grid,int64_t M,int64_t cutoff,int64_t stream){
 int major,dev;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));int bytes=major==9?chunk::dyn<4,64>():chunk::dyn<3,64>();
 if(first)chunk::kernel<4,true><<<grid,256,bytes,(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,(int)cutoff);
 else chunk::kernel<4,false><<<grid,256,bytes,(cudaStream_t)stream>>>(BR_ARGS(v),(int)M,(int)cutoff);
 BR_CUDA_CHECK(cudaGetLastError());}
'''
 return cu

def main():
 parent.prepare_parent();base=BASE.read_text();assert hashlib.sha256(base.encode()).hexdigest()==parent.SHA
 cu=cuda(base);cpp=value(base,'NF_CPP')+'''void bal_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void bal_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> ch_info(int64_t,int64_t);'''
 fs=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','bal_route','bal_run','ch_info']
 plan=function(base,'nf_plan').replace('def nf_plan(data,base,native,streams):','def balance_plan(data,base,native,streams,arm):').replace('mod=nf_module()','mod=balance_module()')
 plan=once(plan,'    p13,p2,pn=', '    low=torch.empty_like(c);nv[6]=low\n    p13,p2,pn=')
 plan=once(plan,'mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)', "mod.bal_route(rp,m,start,low.data_ptr(),{'B8':8,'B4':4,'B2':2,'B0':0,'ONE':0}[arm],torch.cuda.current_stream().cuda_stream)")
 plan=once(plan,'mod.br_run(p13,26,grid,m,side.cuda_stream);mod.br_run(p2,21,grid,m,side.cuda_stream)', "mod.bal_run(p13,1,sms*4,m,0 if arm=='ONE' else 16,side.cuda_stream);mod.bal_run(p2,0,sms*4,m,0 if arm=='ONE' else 16,side.cuda_stream)")
 plan=once(plan,'mod.br_run(pl13,70,sms*light_grid,m,light.cuda_stream);mod.br_run(pl2,71,sms*light_grid,m,light.cuda_stream);ljoin.record()', "if arm!='ONE':\n                mod.br_run(pl13,72,sms*light_grid,m,light.cuda_stream);mod.br_run(pl2,73,sms*light_grid,m,light.cuda_stream)\n            ljoin.record()")
 plan=once(plan,'        mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)', "        if arm not in ('B0','ONE'):\n            mod.pmz_run(pn,1,8,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,8,m,stages,grids[0],parent.cuda_stream)")
 cell='BALANCE_CUDA='+repr(cu)+'\nBALANCE_CPP='+repr(cpp)+'\nBALANCE_FUNCTIONS='+repr(fs)+'\n_BALANCE_MODULE=[]\n'
 cell+='''def balance_module():
 if not _BALANCE_MODULE:
  arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  mod=load_inline(name='p3b_balance_'+hashlib.sha256(BALANCE_CUDA.encode()).hexdigest()[:12],cpp_sources=BALANCE_CPP,cuda_sources=BALANCE_CUDA,functions=BALANCE_FUNCTIONS,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
  mod.br_init();NF_OPTS['pmz']=mod.pmz_init();mod.ch_info(4,0);mod.ch_info(4,1);_BALANCE_MODULE.append(mod)
 return _BALANCE_MODULE[0]
'''+plan+'\n'
 (H/'balance_cell.py').write_text(cell);(H/'balance.cu').write_text(cu)
 out=ROOT/'runs/screen/p3b_balance';out.mkdir(parents=True,exist_ok=True);manifest={}
 for arm in ARMS[2:]:
  hook=f'''
_balance_original_nf_call=nf_call
def nf_call(data,state):
 for fn in balance_plan(data,state['s5_hh_base'],state['moe_bank'],state['nf_streams'],{arm!r})['steps']:out=fn()
 return out
_balance_original_s4=_s4_prepare_banks
def _s4_prepare_banks(state,weights):
 _balance_original_s4(state,weights);balance_module();return state
'''
  candidate=base+'\n'+cell+hook;p=out/(arm.lower()+'.py');p.write_text(candidate)
  patch=''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+str(BASE.relative_to(ROOT)),tofile='b/'+str(BASE.relative_to(ROOT))))
  (H/(arm.lower()+'.patch')).write_text(patch)
  manifest[arm]=dict(candidate=str(p.relative_to(ROOT)),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest())
 manifest.update(base_sha256=parent.SHA,cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),arms=ARMS)
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Exact synthetic helpers and S15 control, already admitted in the preceding batch.
 packed=(B/'s15_review/s15_review_payload.py').read_text();blob=next(n for n in ast.parse(packed).body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)).value.args[0].args[0].args[0].args[0].value
 src=zlib.decompress(base64.b85decode(blob)).decode();core=src[:src.index('REVIEW_CELLS=')]
 chcell=parent.source('CHUNK4')[1].split('_p3b_original_s4_prepare_banks=')[0]
 oracle=(H/'oracle.cu.inc').read_text()
 body=core+'\nBALANCE_CELL='+repr(cell)+'\nCHUNK_CELL='+repr(chcell)+'\nCLOCK_S15='+repr(function(core,'nf_plan'))+'\nCLOCK_CH4='+repr(function(chcell,'p3b_plan'))+'\nORACLE_CUDA='+repr(oracle)+'\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text()
 compile(body,'<p3b-balance>','exec');encoded=base64.b85encode(zlib.compress(body.encode(),9)).decode()
 payload='import base64,zlib\nexec(compile(zlib.decompress(base64.b85decode('+repr(encoded)+')),"<p3b-balance>","exec"))\ndef lab(deadline,weights,state=None):return balance_lab(deadline,weights,state)\n'
 (H/'balance_payload.py').write_text(payload);(H/'expanded.py').write_text(body)
 (H/'payload.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),expanded_sha256=hashlib.sha256(body.encode()).hexdigest(),bytes=len(payload)),indent=2)+'\n')
 print(manifest)

if __name__=='__main__':main()
