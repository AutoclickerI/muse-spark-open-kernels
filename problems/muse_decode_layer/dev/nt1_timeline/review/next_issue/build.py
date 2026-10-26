"""One change per arm on the measured per-M SHARD2/PAIR/PAIR+BLOCK parent."""
import ast,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;V=D.parent;R=D.parents[5];sys.path.insert(0,str(V))
_spec=importlib.util.spec_from_file_location('p3a_review_builder',V/'build.py');review=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(review)
PARENTS=[('SHARD2',),('PAIR',),('PAIR','BLOCK')]
MS=[(1,),(8,),(16,32)]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def rename(cell,tag):
 return cell.replace('smallm_',tag+'_').replace('_SMALLM_MODULE','_'+tag.upper()+'_MODULE').replace('SMALLM_FLAGS_BUILT',tag.upper()+'_FLAGS_BUILT').replace('SMALLM_VAR_',tag.upper()+'_')
def parent(features):
 v=review.source(features[0],rename=False)
 for arm in features[1:]:v=review.source(arm,initial=v,rename=False)
 return v
FAST_HELPER='''__device__ __forceinline__ float cancel_norm(const float* meta,int e,float sum){
 const float4 p=__ldg(reinterpret_cast<const float4*>(meta)+e);
 const float energy=fmaf(sum,p.x,p.y);
 return sum==0.f?0.f:p.z*rsqrtf(energy);
}
'''
FAST_BIND='''    a=state['gs2'].double()*256.0
    q=a*1e4;large=q.abs()>=1.0;qq=q*q
    mul=torch.where(large,torch.ones_like(q),qq)
    eps=torch.where(large,4096.0/qq,torch.full_like(q,4096.0))
    num=torch.where(large,64.0*q.sign(),64.0*q)
    state['cancel_meta']=torch.stack((mul.float(),eps.float(),num.float(),state['gs2']),dim=1).contiguous()
'''
def change(arm,initial):
 cell,cuda=initial
 if arm=='FAST':
  cell,cuda=review.source('CANCEL',initial=initial,rename=False);old=cuda
  a=cuda.index('__device__ __forceinline__ float cancel_norm');b=cuda.index('\nstruct PrefetchState',a)
  cuda=cuda[:a]+FAST_HELPER+cuda[b:]
  before="    a=state['gs2'].double()*256.0\n    state['cancel_meta']=torch.cat((state['gs2'],(1e-8/(a*a)).float(),a.sign().float())).contiguous()\n"
  assert cell.count(before)==1;cell=cell.replace(before,FAST_BIND)
 elif arm=='INDEX':
  old=cuda;a=cuda.index('#pragma unroll\n      for (int j = 0; j < NT; ++j) {',cuda.index('auto unit_body'));b=cuda.index('    }\n    if constexpr (TRACE)',a)
  cuda=cuda[:a]+(D/'bfill.py.inc').read_text()+cuda[b:]
 elif arm=='FLUSH1':
  old=cuda
  before='  if constexpr(VNR)if(cur_pass==1)flush_vec_norm();\n      if constexpr(VNR)if(cur_pass==1)flush_vec_norm();'
  assert cuda.count(before)==1;cuda=cuda.replace(before,'      if constexpr(VNR)if(cur_pass==1)flush_vec_norm();')
 elif arm=='BASE':return cell,cuda
 else:raise ValueError(arm)
 assert repr(old) in cell;cell=cell.replace(repr(old),repr(cuda),1);ast.parse(cell);return cell,cuda
def main():
 arms=sys.argv[1:] or ['BASE','FAST','INDEX','FLUSH1'];base,_,_=review.original();b=base.index('def _s4_prepare_banks(');manifest={};components={};screen={}
 out=R/'runs/screen/p3a_next_issue';out.mkdir(exist_ok=True,parents=True)
 for arm in arms:
  cells=[];variants=[]
  for i,features in enumerate(PARENTS):
   cell,cu=change(arm,parent(features));tag='rvc_n'+arm.lower()+str(i);cell=rename(cell,tag);name=arm+str(i);p=D/(name.lower()+'_cell.py');p.write_text(cell);(out/(name.lower()+'.cu')).write_text(cu)
   spec=dict(cell=str(p.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),prefix=tag,features=list(features)+([] if arm=='BASE' else [arm]),M=list(MS[i]))
   components[name]=spec;variants.append(spec);cells.append(cell)
  tag='rvc_n'+arm.lower()
  selector=f'\ndef {tag}_module():\n'+''.join(f'    {s["prefix"]}_module()\n' for s in variants)
  selector+=f'def {tag}_bind(state,flags_by_m):\n'+''.join(f'    {s["prefix"]}_bind(state,flags_by_m)\n' for s in variants)+'    return state\n'
  selector+=f"def {tag}_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n"+''.join(f'    if m in {tuple(s["M"])}:return {s["prefix"]}_experts(data,state,flags,stages,grid)\n' for s in variants)+'    return _next_original_experts(data,state,flags,stages,grid)\n'
  aliases='\n_next_original_bind=smallm_bind\n_next_original_experts=smallm_experts\n'
  wrapper=f'''\ndef smallm_bind(state,flags_by_m):
    _next_original_bind(state,flags_by_m)
    return {tag}_bind(state,flags_by_m)
def smallm_experts(data,state,flags,stages=2,grid=264):
    return {tag}_experts(data,state,flags,stages,grid)
'''
  cand=base[:b]+'\n'.join(cells)+aliases+selector+wrapper+'\n'+base[b:];ast.parse(cand);cp=out/(arm.lower()+'.py');cp.write_text(cand)
  patch=''.join(difflib.unified_diff(base.splitlines(True),cand.splitlines(True),fromfile='a/'+str(review.BASE.relative_to(R)),tofile='b/'+str(review.BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch)
  (D/(arm.lower()+'_selector.py.inc')).write_text(selector)
  manifest[arm]=dict(base_sha256=review.BASE_SHA,parent='measured c49 per-M composition',candidate=str(cp.relative_to(R)),candidate_sha256=sha(cand),patch=str(pp.relative_to(R)),patch_sha256=sha(patch),prefix=tag,variants=variants)
  if arm!='BASE':screen[arm]=dict(candidate=manifest[arm]['candidate'],candidate_sha256=sha(cand))
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(D/'components_manifest.json').write_text(json.dumps(components,indent=2)+'\n');(D/'screen_manifest.json').write_text(json.dumps(screen,indent=2)+'\n')
 print({a:m['candidate_sha256'] for a,m in manifest.items()})
if __name__=='__main__':main()
