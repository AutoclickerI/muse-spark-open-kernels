"""Ready structural decoder: bank-replicated byte table, one PRMT forms each shared address."""
import ast,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;S=D.parent;P=S.parent;R=D.parents[5];sys.path.insert(0,str(P/'review'))
sp=importlib.util.spec_from_file_location('dq_review',P/'review/build.py');review=importlib.util.module_from_spec(sp);sp.loader.exec_module(review)
def source(block=False):
 cell,cu=review.source('PAIR',rename=False)
 if block:cell,cu=review.source('BLOCK',initial=(cell,cu),rename=False)
 old=cu
 anchor='  extern __shared__ __align__(128) uint8_t smem[];'
 assert cu.count(anchor)==1
 cu=cu.replace(anchor,'''  __shared__ __align__(256) uint32_t dqtable[256*64];
  for(int i=threadIdx.x;i<256*64;i+=NTHR){uint32_t u=i>>6;dqtable[i]=((u&7)<<6)|((u&8)<<12)|((u&112)<<18)|((u&128)<<24);}
  __syncthreads();
'''+anchor)
 anchor='''        uint32_t o[16];
        dq8(wv.x, wv.y, o);
        dq8(wv.z, wv.w, o + 8);'''
 assert anchor in cu
 cu=cu.replace(anchor,'''        uint32_t o[16];
        const uint32_t tableptr=smem_u32(dqtable)+lane*4;
#pragma unroll
        for(int k=0;k<16;++k){const uint32_t z=k<4?wv.x:k<8?wv.y:k<12?wv.z:wv.w;
          const uint32_t ptr=__byte_perm(z,tableptr,0x7604u+((k&3)<<4))+(tableptr&0xff00u);
          asm volatile("ld.shared.u32 %0,[%1];":"=r"(o[k]):"r"(ptr));}
''')
 assert repr(old) in cell;cell=cell.replace(repr(old),repr(cu),1)
 before="    return state\n\n\ndef smallm_experts"
 assert before in cell
 cell=cell.replace(before,"    if 'lut' not in state['banks']:\n        state['banks']['lut']={k:plut_pack(v) for k,v in state['banks']['lop3'].items()}\n"+before)
 cell=cell.replace("bank = state['banks']['lop3']","bank = state['banks']['lut']")
 # Reuse the proven exact4-bit code-pair repacker; only lookup/layout changes.
 pack=(P/'plut_cell.py').read_text();cell+='\n'+pack[pack.index('import triton'):];ast.parse(cell)
 return cell,cu
if __name__=='__main__':
 out=R/'runs/screen/p3a_structural';cells={};manifest={};base,_,_=review.original();b=base.index('def _s4_prepare_banks(')
 for block in (False,True):
  cell,cu=source(block);name='lut64'+str(int(block));tag='rvc_dq'+str(int(block));cell=cell.replace('smallm_',tag+'_').replace('_SMALLM_MODULE','_'+tag.upper()+'_MODULE').replace('SMALLM_FLAGS_BUILT',tag.upper()+'_FLAGS_BUILT').replace('SMALLM_VAR_',tag.upper()+'_');(out/(name+'.cu')).write_text(cu);(D/(name+'_cell.py')).write_text(cell)
  cells[name]=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),cell=str((D/(name+'_cell.py')).relative_to(R)),prefix=tag)
 for depth in (1,2):
  arm='LUT'+str(depth);tag='rvc_dq_lut'+str(depth);specs=list(cells.values())
  selector=f'\ndef {tag}_module():\n    rvc_dq0_module()\n    rvc_dq1_module()\ndef {tag}_bind(state,flags_by_m):\n    rvc_dq0_bind(state,{{8:69}})\n    rvc_dq1_bind(state,{{8:69}})\n    return state\n'
  selector+=f"def {tag}_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n    if m==1:return _dq_original_experts(data,state,flags,stages,grid)\n    fn=rvc_dq0_experts if m<=8 else rvc_dq1_experts\n    return fn(data,state,flags,{depth},grid)\n"
  wrapper=f'\ndef smallm_bind(state,flags_by_m):\n    _dq_original_bind(state,flags_by_m)\n    return {tag}_bind(state,flags_by_m)\ndef smallm_experts(data,state,flags,stages=2,grid=264):\n    return {tag}_experts(data,state,flags,stages,grid)\n'
  text=base[:b]+'\n'.join((R/s['cell']).read_text() for s in specs)+'\n_dq_original_bind=smallm_bind\n_dq_original_experts=smallm_experts\n'+selector+wrapper+'\n'+base[b:];ast.parse(text);p=out/(arm.lower()+'.py');p.write_text(text);patch=''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='a/'+str(review.BASE.relative_to(R)),tofile='b/'+str(review.BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch);(D/(arm.lower()+'_selector.py.inc')).write_text(selector)
  manifest[arm]=dict(base_sha256=review.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),patch=str(pp.relative_to(R)),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),prefix=tag,depth=depth)
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');print(manifest)
