"""Structural depth and phase-boundary arms on the measured PAIR(+BLOCK) parent."""
import ast,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;R=D.parents[4];sys.path.insert(0,str(P/'review'))
s=importlib.util.spec_from_file_location('struct_review',P/'review/build.py');review=importlib.util.module_from_spec(s);s.loader.exec_module(review)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def rename(cell,tag):return cell.replace('smallm_',tag+'_').replace('_SMALLM_MODULE','_'+tag.upper()+'_MODULE').replace('SMALLM_FLAGS_BUILT',tag.upper()+'_FLAGS_BUILT').replace('SMALLM_VAR_',tag.upper()+'_')
def parent(block):
 v=review.source('PAIR',rename=False)
 return review.source('BLOCK',initial=v,rename=False) if block else v
def split_source(cell,cuda):
 old=cuda
 def sub(a,b,n=1):
  nonlocal cuda
  assert cuda.count(a)==n,(a,cuda.count(a),n);cuda=cuda.replace(a,b)
 sub('template <int NT, int F>\n__global__','template <int NT, int F, int MODE=0>\n__global__')
 sub('if constexpr (PDL) asm volatile("griddepcontrol.wait;', 'if constexpr (PDL && MODE<=1) asm volatile("griddepcontrol.wait;')
 # MODE1/2 run one independent phase; MODE3 rebuilds routes only and combines.
 # No value-derived preparation: routes and every activation remain inside timed calls.
 sub('    first_count = b1 - a1; total_count = first_count + b2 - a2;','    if constexpr(MODE==1){a2=b2=0;}\n    if constexpr(MODE==2){a1=b1=0;}\n    first_count = b1 - a1; total_count = first_count + b2 - a2;',3)
 anchor='''    if (tid == 0) {
      if (total_count) {'''
 sub(anchor,'''    if (tid == 0 && MODE!=3) {
      if (total_count) {''')
 # The M1 SLOT special case is never dispatched to split modes, but avoid issuing its copies.
 sub('prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);','if constexpr(MODE!=3)prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);',2)
 sub('if constexpr (SLOT) if constexpr(MODE!=3)prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);\n      else prefetch_one','if constexpr (SLOT) {if constexpr(MODE!=3)prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);}\n      else prefetch_one')
 # Keep all full-kernel fences/publications unchanged; split kernels use ordered CUDA launches.
 sub('if (cur_pass == 0 && seg_tiles > 0) {','if (MODE==0 && cur_pass == 0 && seg_tiles > 0) {',2)
 sub('while (ld_acquire(cnt + C_DONE + e'+('*4+nb' if 'e*4+nb' in cuda else '')+') < target) __nanosleep(64);','if constexpr(MODE==0)while (ld_acquire(cnt + C_DONE + e'+('*4+nb' if 'e*4+nb' in cuda else '')+') < target) __nanosleep(64);')
 a=cuda.index('  for (int pass = 0; pass < 2; ++pass) {',cuda.index('auto unit_body'))
 b=cuda.index('  // ---------------- grid barrier',a)
 cuda=cuda[:a]+'  if constexpr(MODE!=3){\n'+cuda[a:b]+'''  }
  if constexpr(MODE==1||MODE==2){__threadfence();bar_consumers();return;}
'''+cuda[b:]
 a=cuda.index('  // ---------------- grid barrier');b=cuda.index('  // ---------------- combine',a)
 cuda=cuda[:a]+'  if constexpr(MODE==0){\n'+cuda[a:b]+'  }\n'+cuda[b:]
 # Additional entrypoints, with exact mode-specific occupancy and shared allocation.
 extra='''
template<int MODE>void structural_launch(std::vector<int64_t>&v,int M,int start,int stages,int grid,cudaStream_t stream){
 constexpr int F=69;
 int resident=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,moe_var<1,F,MODE>,NTHR,smem_bytes(1,stages,M)));
 grid=MODE==3?2*M:std::min(grid,std::max(1,resident)*132);
 cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(NTHR);cfg.dynamicSmemBytes=smem_bytes(1,stages,M);cfg.stream=stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;
 if constexpr(MODE==1){cfg.attrs=&attr;cfg.numAttrs=1;}
 C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,moe_var<1,F,MODE>,
 (const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],
 (const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],
 (__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages,(uint64_t*)0));
}
void structural_run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream,int64_t flags){
 if(M<2||M>32||flags!=69)throw std::runtime_error("structural split requires M2..32 flags69");
 structural_launch<1>(v,M,start,stages,grid,(cudaStream_t)stream);
 structural_launch<2>(v,M,start,stages,grid,(cudaStream_t)stream);
 structural_launch<3>(v,M,start,0,2*M,(cudaStream_t)stream);
}
std::vector<int64_t> structural_info(int64_t mode,int64_t stages,int64_t M){int r=0;cudaFuncAttributes a{};
#define MODE_INFO(K) if(mode==K){C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_var<1,69,K>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r,moe_var<1,69,K>,NTHR,smem_bytes(1,stages,M)));}
 MODE_INFO(1) MODE_INFO(2) MODE_INFO(3)
#undef MODE_INFO
 return{r,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)smem_bytes(1,stages,M)};}
'''
 cuda+='\n'+extra
 anchor='  VAR_FLAGS(SET_ATTR)'
 sub(anchor,'''  VAR_FLAGS(SET_ATTR)
#define SPLIT_ATTR(K) {cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_var<1,69,K>));C10_CUDA_CHECK(cudaFuncSetAttribute(moe_var<1,69,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,optin-(int)a.sharedSizeBytes));}
 SPLIT_ATTR(1) SPLIT_ATTR(2) SPLIT_ATTR(3)
#undef SPLIT_ATTR''')
 assert repr(old) in cell;cell=cell.replace(repr(old),repr(cuda),1)
 cell=cell.replace("functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run', 'epi_info']", "functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run', 'epi_info', 'structural_run', 'structural_info']")
 # CPP declarations are separate from the CUDA literal.
 cell=cell.replace('std::vector<int64_t> epi_info(int64_t,int64_t,int64_t);','std::vector<int64_t> epi_info(int64_t,int64_t,int64_t);\nvoid structural_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\nstd::vector<int64_t> structural_info(int64_t,int64_t,int64_t);')
 cell=cell.replace("grid = min(grid, state['sms'] * mod.epi_info(flags, stages, m)[0])", "grid = min(grid,state['sms']*max(1,mod.structural_info(1,stages,m)[0]))")
 cell=cell.replace('mod.run(ptrs, m,','mod.structural_run(ptrs, m,')
 # Fixed H200 SM count is unnecessary; query the actual device in the host launcher.
 cuda=cuda.replace('grid=MODE==3?2*M:std::min(grid,std::max(1,resident)*132);','int dev,sms;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,dev));\n grid=MODE==3?2*M:std::min(grid,std::max(1,resident)*sms);')
 prior=ast.literal_eval(next(n.value for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA'))
 cell=cell.replace(repr(prior),repr(cuda),1);ast.parse(cell)
 return cell,cuda
def main():
 base,_,_=review.original();b=base.index('def _s4_prepare_banks(');out=R/'runs/screen/p3a_structural';out.mkdir(parents=True,exist_ok=True);cells={};manifest={}
 for block in (False,True):
  for split in (False,True):
   cell,cu=parent(block)
   if split:cell,cu=split_source(cell,cu)
   name=('SPLIT' if split else 'FUSED')+str(int(block));tag='rvc_struct_'+name.lower();cell=rename(cell,tag);cp=D/(name.lower()+'_cell.py');cp.write_text(cell);(out/(name.lower()+'.cu')).write_text(cu)
   cells[name]=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),prefix=tag,split=split,block=block)
 arms={'BASE':(False,2),'D3':(False,3),'D4':(False,4),'SPLIT2':(True,2),'SPLIT3':(True,3),'SPLIT4':(True,4)}
 for arm,(split,depth) in arms.items():
  specs=[cells[('SPLIT' if split else 'FUSED')+str(i)] for i in (0,1)];tag='rvc_struct_'+arm.lower()
  # Fused selector shares its tag with the FUSED cell only when arm names differ.
  selector=f'\ndef {tag}_module():\n'+''.join(f'    {s["prefix"]}_module()\n' for s in specs)
  selector+=f'def {tag}_bind(state,flags_by_m):\n'+''.join(f'    {s["prefix"]}_bind(state,{{8:69}})\n' for s in specs)+'    return state\n'
  selector+=f"def {tag}_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n    if m==1:return _struct_original_experts(data,state,flags,stages,grid)\n    fn={specs[0]['prefix']}_experts if m<=8 else {specs[1]['prefix']}_experts\n    return fn(data,state,flags,{depth},grid)\n"
  wrapper=f'''\ndef smallm_bind(state,flags_by_m):
    _struct_original_bind(state,flags_by_m)
    return {tag}_bind(state,flags_by_m)
def smallm_experts(data,state,flags,stages=2,grid=264):
    return {tag}_experts(data,state,flags,stages,grid)
'''
  text=base[:b]+'\n'.join((R/s['cell']).read_text() for s in specs)+'\n_struct_original_bind=smallm_bind\n_struct_original_experts=smallm_experts\n'+selector+wrapper+'\n'+base[b:];ast.parse(text);p=out/(arm.lower()+'.py');p.write_text(text)
  patch=''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='a/'+str(review.BASE.relative_to(R)),tofile='b/'+str(review.BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch);(D/(arm.lower()+'_selector.py.inc')).write_text(selector)
  manifest[arm]=dict(base_sha256=review.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=sha(text),patch=str(pp.relative_to(R)),patch_sha256=sha(patch),cells=specs,depth=depth,split=split,prefix=tag)
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'screen_manifest.json').write_text(json.dumps({a:dict(candidate=m['candidate'],candidate_sha256=m['candidate_sha256']) for a,m in manifest.items() if a!='BASE'},indent=2)+'\n');print({a:m['candidate_sha256'] for a,m in manifest.items()})
if __name__=='__main__':main()
