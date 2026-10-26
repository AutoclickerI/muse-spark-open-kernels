"""Exact S19 wait-removal oracle and producer-owned W13/W2 slice candidates."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_dependency';O.mkdir(exist_ok=True,parents=True)
sp=importlib.util.spec_from_file_location('ipbase',D.parent/'intpipe/build.py');base=importlib.util.module_from_spec(sp);sp.loader.exec_module(base);sha=base.sha

def oracle(cu):
 import re
 pattern=r'while \(ld_acquire\(cnt \+ C_DONE \+ e(?:\*4\+nb)?\) < target\) __nanosleep\(64\);'
 cu,n=re.subn(pattern,lambda m:m[0].replace('while (','(void)(').replace(' < target) __nanosleep(64);',');'),cu);assert n==1
 before='while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);';assert cu.count(before)==1;cu=cu.replace(before,'(void)ld_acquire(cnt+C_GEN);')
 return cu

def main():
 s=base.original();nodes=ast.parse(s).body;cells={}
 for i,prefix in enumerate(base.PREFIXES[1:],1):
  c=base.cell(s,prefix);cu=next(ast.literal_eval(n.value) for n in ast.parse(c).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'))
  for kind in ('BASE','NOWAIT'):
   new=cu if kind=='BASE' else oracle(cu);tag='dp_'+kind.lower()+str(i);text=c.replace(repr(cu),repr(new),1).replace(prefix,tag).replace(prefix.upper(),tag.upper());p=D/(kind.lower()+str(i)+'_cell.py');p.write_text(text);(O/(kind.lower()+str(i)+'.cu')).write_text(new);cells[kind+str(i)]=dict(cell=str(p.relative_to(R)),cell_sha256=sha(text),cuda_sha256=sha(new),prefix=tag,kind=kind)
 header=cu[:cu.index('__device__ __forceinline__ void prefetch_one(')]
 cuda=header+(D/'slice_body.cu').read_text();(O/'slice.cu').write_text(cuda)
 cpp='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
void dep_init();std::vector<int64_t> dep_info(int,int);void dep_run(std::vector<int64_t>,int,int,int,int,int,int64_t);
'''
 text='import hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nDEP_CUDA='+repr(cuda)+'\nDEP_CPP='+repr(cpp)+'\n'+(D/'slice_api.py.inc').read_text();p=D/'slice_cell.py';p.write_text(text);cells['SLICE']=dict(cell=str(p.relative_to(R)),cell_sha256=sha(text),cuda_sha256=sha(cuda),prefix='dep',kind='SLICE')
 # Same72KiB raw capacity in W2 despite its smaller copies;old two-slot arm stays a control.
 deep=(D/'slice_body.cu').read_text().replace('2*TILE_BYTES+8192+32+sizeof(Route)', '2*TILE_BYTES+8192+2*(2*TILE_BYTES/(8*(S/64)*576))*8+sizeof(Route)')
 deep=deep.replace('constexpr int NS=4096/S,GP=S/64,W2W=8*GP*512,W2B=W2W+8*32*GP*2;', 'constexpr int NS=4096/S,GP=S/64,W2W=8*GP*512,W2B=W2W+8*32*GP*2,WD=2*TILE_BYTES/W2B;')
 deep=deep.replace('uint64_t* empty=full+2;', 'uint64_t* empty=full+WD;').replace('Route* rt=reinterpret_cast<Route*>(empty+2);','Route* rt=reinterpret_cast<Route*>(empty+WD);')
 a=deep.index('  { // W2:');b=deep.index('  __syncthreads(); // All local readers',a);block=deep[a:b]
 block=block.replace('s<2;++s','s<WD;++s').replace('int st=it&1,ph=(it>>1)&1;','int st=it%WD,ph=(it/WD)&1;').replace('it+2<32','it+WD<32').replace('int next=it+2;int st2=next&1,ph2=(next>>1)&1;','int next=it+WD;int st2=next%WD,ph2=(next/WD)&1;')
 deep=deep[:a]+block+deep[b:];deep=deep.replace(' GO(64,0)','').replace(' INFO(64,0)','').replace(' ATTR(64,0)','')
 deep_cuda=(header+deep).replace('dep_run','dpx_run').replace('dep_info','dpx_info').replace('dep_init','dpx_init');(O/'deep.cu').write_text(deep_cuda)
 deep_cpp=cpp.replace('dep_run','dpx_run').replace('dep_info','dpx_info').replace('dep_init','dpx_init')
 api=(D/'slice_api.py.inc').read_text().replace('dep_','dpx_').replace('DEP_','DPX_')
 deep_text='import hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nDPX_CUDA='+repr(deep_cuda)+'\nDPX_CPP='+repr(deep_cpp)+'\n'+api
 p=D/'deep_cell.py';p.write_text(deep_text);cells['DEEP']=dict(cell=str(p.relative_to(R)),cell_sha256=sha(deep_text),cuda_sha256=sha(deep_cuda),prefix='dpx',kind='DEEP')
 manifest={}
 for arm,S,mode in (('A64',64,0),('A128',128,0),('P128',128,1),('A128D',128,0),('P128D',128,1)):
  prefix='dpx' if arm.endswith('D') else 'dep'
  extra='\n'+(deep_text if arm.endswith('D') else text)+f'''
_dep_s19_experts=smallm_experts
_dep_s19_banks=_s4_prepare_banks
def _s4_prepare_banks(state,weights):
    _dep_s19_banks(state,weights)
    {prefix}_bind(state['s4_nt1_bank'],{S})
def smallm_experts(data,state,flags,stages=2,grid=264):
    if data['hidden_states'].shape[0] in (8,16,32):return {prefix}_experts(data,state,{S},{mode},grid)
    return _dep_s19_experts(data,state,flags,stages,grid)
'''
  cand=s+extra;ast.parse(cand);p=O/(arm.lower()+'.py');p.write_text(cand);patch=''.join(difflib.unified_diff(s.splitlines(True),cand.splitlines(True),fromfile='a/'+str(base.BASE.relative_to(R)),tofile='b/'+str(base.BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch)
  manifest[arm]=dict(base=str(base.BASE.relative_to(R)),base_sha256=base.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=sha(cand),patch=str(pp.relative_to(R)),patch_sha256=sha(patch),S=S,mode=mode,prefix=prefix)
 (D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built',list(manifest),'oracle',cells['NOWAIT1']['cuda_sha256'][:12])
if __name__=='__main__':main()
