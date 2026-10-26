"""Three full-slot layouts: phase-local scratch and register-safe reduction storage."""
import ast,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;S=D.parent;P=S.parent;R=D.parents[5]
sp=importlib.util.spec_from_file_location('occ_parent',S/'build.py');parent=importlib.util.module_from_spec(sp);sp.loader.exec_module(parent)
sha=parent.sha
KINDS=('SINGLE','HALF','DOUBLE')
def source(kind,block):
 cell,cu=parent.parent(block);old=cu;alias=kind in ('SINGLE','DOUBLE');half=kind in ('HALF','DOUBLE');redbuf=1 if kind in ('SINGLE','HALF') else 2
 def sub(a,b,n=1):
  nonlocal cu
  assert cu.count(a)==n,(kind,a,cu.count(a));cu=cu.replace(a,b)
 sub('  constexpr int RED = NT * 4;','  constexpr int RED = NT * 4;\n  constexpr int RWC=VEC?'+('NCW/2' if half else 'NCW')+':NCW;\n  constexpr int RBUF=VEC?'+str(redbuf)+':2;')
 sub('red + 2 * NCW * 32 * RED','red + RBUF * RWC * 32 * RED')
 if alias:
  sub('  int run[NLOC];\n  int wc[NCW][NLOC];\n','')
  sub('  int off[NLOC];','  uint16_t off[NLOC];')
  sub('  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);','  int* run=reinterpret_cast<int*>(red);\n  int (*wc)[NLOC]=reinterpret_cast<int(*)[NLOC]>(run+NLOC);\n  int8_t* eflat=reinterpret_cast<int8_t*>(wc+NCW);')
  sub('  int16_t* pair_of = pos_of + npairs;','  uint8_t* pair_of = reinterpret_cast<uint8_t*>(rt+1);')
  cu=cu.replace('rt->wc','wc').replace('rt->run','run')
  sub('      if (lane < 8) pos_of[lane] = static_cast<int16_t>(e >= 0 ? r : -1);\n','')
  sub('          pos_of[i] = static_cast<int16_t>(pos);\n','')
  sub('        } else {\n          pos_of[i] = -1;\n        }','        }')
  sub('static_cast<int16_t>(lane)','static_cast<uint8_t>(lane)')
  sub('static_cast<int16_t>(i)','static_cast<uint8_t>(i)')
  anchor='  // ---------------- grid barrier (or last-arriver election) ----------------'
  sub(anchor,'''  // Reduction scratch is dead. Rebuild the inverse route here,inside the timed call.
  bar_consumers();
  for(int i=tid;i<npairs;i+=NTHR)pos_of[i]=-1;
  bar_consumers();
  const int route_pairs=SLOT?rt->P:rt->off[NLOC-1]+rt->n[NLOC-1];
  for(int p=tid;p<route_pairs;p+=NTHR)pos_of[pair_of[p]]=static_cast<int16_t>(p);
  bar_consumers();
'''+anchor)
 else:
  sub('  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);','  int8_t* eflat = reinterpret_cast<int8_t*>(red);')
  sub('  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));','  int16_t* pos_of = reinterpret_cast<int16_t*>(rt + 1);')
 # All route-only scratch aliases reduction storage; establish its final read before first reduction write.
 sub('  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1]', '  bar_consumers(); // route scratch lifetime ends before reduction starts\n  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1]')
 if half:
  anchor='''    if constexpr(VEC){float* p=red+(buf*NCW+warp)*128;
      *reinterpret_cast<float2*>(p+lane*2)=make_float2(acc[0][0],acc[0][2]);
      *reinterpret_cast<float2*>(p+64+lane*2)=make_float2(acc[0][1],acc[0][3]);
    }else{'''
  sub(anchor,'''    if constexpr(VEC){float* p=red+(buf*RWC+(warp&3))*128;
      if(warp<4){
        *reinterpret_cast<float2*>(p+lane*2)=make_float2(acc[0][0],acc[0][2]);
        *reinterpret_cast<float2*>(p+64+lane*2)=make_float2(acc[0][1],acc[0][3]);
      }
      bar_consumers();
      if(warp>=4){
        float2 a=*reinterpret_cast<const float2*>(p+lane*2);
        float2 b=*reinterpret_cast<const float2*>(p+64+lane*2);
        a.x+=acc[0][0];a.y+=acc[0][2];b.x+=acc[0][1];b.y+=acc[0][3];
        *reinterpret_cast<float2*>(p+lane*2)=a;
        *reinterpret_cast<float2*>(p+64+lane*2)=b;
      }
    }else{''')
  sub('for(int w=0;w<NCW;++w){float2 v=', 'for(int w=0;w<RWC;++w){float2 v=')
 else:sub('red+(buf*NCW+warp)*128','red+(buf*RWC+warp)*128')
 sub('red + buf * NCW * 32 * RED','red + buf * RWC * 32 * RED')
 if redbuf==1:sub('    buf ^= 1;','    if constexpr(VEC)bar_consumers();else buf^=1; // one buffer: every read must finish before reuse')
 # Native NT2 probes keep their original2 full reduction buffers. Only NT1 is dispatched here.
 factor='(nt==1?'+str(redbuf*(4 if half else 8))+':2*NCW)'
 sub('2 * NCW * 32 * nt * 4 * sizeof(float)',factor+' * 32 * nt * 4 * sizeof(float)')
 tail='sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs'
 sub(tail,'sizeof(Route) + '+('npairs' if alias else '4 * npairs'))
 # Assert alias live storage fits either reduction region at all supported M,including nearby shapes.
 sub('  const int npairs = M * 8;\n  extern', '  const int npairs = M * 8;\n  extern')
 assert repr(old) in cell;cell=cell.replace(repr(old),repr(cu),1);ast.parse(cell)
 return cell,cu
if __name__=='__main__':
 out=R/'runs/screen/p3a_occupancy';cells={};manifest={};base,_,_=parent.review.original();b=base.index('def _s4_prepare_banks(')
 for kind in KINDS:
  for block in (False,True):
   cell,cu=source(kind,block);name=kind+str(int(block));tag='rvc_occ_'+name.lower();cell=parent.rename(cell,tag);cp=D/(name.lower()+'_cell.py');cp.write_text(cell);(out/(name.lower()+'.cu')).write_text(cu)
   cells[name]=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),prefix=tag,kind=kind,block=block)
  for depth in (2,3):
   arm=kind+str(depth);tag='rvc_occ_'+arm.lower();specs=[cells[kind+str(i)] for i in (0,1)]
   selector=f'\ndef {tag}_module():\n'+''.join(f'    {s["prefix"]}_module()\n' for s in specs)
   selector+=f'def {tag}_bind(state,flags_by_m):\n'+''.join(f'    {s["prefix"]}_bind(state,{{8:69}})\n' for s in specs)+'    return state\n'
   selector+=f"def {tag}_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n    if m==1:return _occ_original_experts(data,state,flags,stages,grid)\n    fn={specs[0]['prefix']}_experts if m<=8 else {specs[1]['prefix']}_experts\n    return fn(data,state,flags,{depth},grid)\n"
   wrapper=f'\ndef smallm_bind(state,flags_by_m):\n    _occ_original_bind(state,flags_by_m)\n    return {tag}_bind(state,flags_by_m)\ndef smallm_experts(data,state,flags,stages=2,grid=264):\n    return {tag}_experts(data,state,flags,stages,grid)\n'
   text=base[:b]+'\n'.join((R/s['cell']).read_text() for s in specs)+'\n_occ_original_bind=smallm_bind\n_occ_original_experts=smallm_experts\n'+selector+wrapper+'\n'+base[b:];ast.parse(text);p=out/(arm.lower()+'.py');p.write_text(text);patch=''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='a/'+str(parent.review.BASE.relative_to(R)),tofile='b/'+str(parent.review.BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch);(D/(arm.lower()+'_selector.py.inc')).write_text(selector)
   manifest[arm]=dict(base_sha256=parent.review.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=sha(text),patch=str(pp.relative_to(R)),patch_sha256=sha(patch),prefix=tag,depth=depth,kind=kind)
 (D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built',list(manifest))
