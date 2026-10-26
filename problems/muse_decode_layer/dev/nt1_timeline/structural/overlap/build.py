"""Independent TMA producer with dynamic registers; timed B-cache only stores BF16 activations."""
import ast,difflib,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;S=D.parent;P=S.parent;R=D.parents[5]
sp=importlib.util.spec_from_file_location('overlap_parent',S/'build.py');parent=importlib.util.module_from_spec(sp);sp.loader.exec_module(parent);sha=parent.sha
from clock_code import instrument
KINDS=('CACHE','WS4','WS8','WS8L2')
def source(kind,block):
 cell,cu=parent.parent(block);old=cu;ws=kind.startswith('WS');four=kind=='WS4';cached=not four
 def sub(a,b,n=1):
  nonlocal cu
  assert cu.count(a)==n,(kind,a,cu.count(a));cu=cu.replace(a,b)
 if four:
  sub('constexpr int NCW = 8;','constexpr int NCW = 4;');cu=cu.replace('for (int g = 0; g < 8; ++g)','for (int g = 0; g < 16; ++g)');sub('uint32_t Bf[NT][8][8]','uint32_t Bf[NT][16][8]');sub('warp * 512 + g * 64','warp * 1024 + g * 64');sub('warp * 8 * 32 + lane','warp * 16 * 32 + lane')
  a=cu.index('      const uint4 scv =');b=cu.index('\n#pragma unroll',a)
  cu=cu[:a]+'''      const uint4 scv=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*64+lane);
      const uint4 scv1=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*64+32+lane);
      const uint32_t scw[8]={scv.x,scv.y,scv.z,scv.w,scv1.x,scv1.y,scv1.z,scv1.w};'''+cu[b:]
  sub('it < 2 * M','it < 4 * M');sub('const int t = it >> 1;','const int t = it >> 2;');sub('const int c0 = (it & 1) * 2048 + tid * 8;','const int c0 = (it & 3) * 1024 + tid * 8;');sub('dump(LAST || c < 2 * M);','dump(LAST || c < 4 * M);')
 if cached:
  sub('  int run[NLOC];\n  int wc[NCW][NLOC];\n','');sub('  int off[NLOC];','  uint16_t off[NLOC];')
  sub('  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);','  uint4* bcache=reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(rt+1)+((npairs+15)&~15));\n  int* run=reinterpret_cast<int*>(bcache);\n  int (*wc)[NLOC]=reinterpret_cast<int(*)[NLOC]>(run+NLOC);\n  int8_t* eflat=reinterpret_cast<int8_t*>(wc+NCW);')
  sub('  int16_t* pair_of = pos_of + npairs;','  uint8_t* pair_of = reinterpret_cast<uint8_t*>(rt+1);');cu=cu.replace('rt->wc','wc').replace('rt->run','run')
  sub('      if (lane < 8) pos_of[lane] = static_cast<int16_t>(e >= 0 ? r : -1);\n','');sub('          pos_of[i] = static_cast<int16_t>(pos);\n','');sub('        } else {\n          pos_of[i] = -1;\n        }','        }');sub('static_cast<int16_t>(lane)','static_cast<uint8_t>(lane)');sub('static_cast<int16_t>(i)','static_cast<uint8_t>(i)')
  anchor='  // ---------------- grid barrier (or last-arriver election) ----------------'
  sub(anchor,'''  bar_consumers();
  for(int i=tid;i<npairs;i+=NCT)pos_of[i]=-1;
  bar_consumers();
  const int route_pairs=SLOT?rt->P:rt->off[NLOC-1]+rt->n[NLOC-1];
  for(int p=tid;p<route_pairs;p+=NCT)pos_of[pair_of[p]]=static_cast<int16_t>(p);
  bar_consumers();
'''+anchor)
  sub('  uint32_t Bf[NT][8][8];','  uint32_t Bf[NT][4][8];')
  assignments='''          Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;
          Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;'''
  sub(assignments,'''          if(g<4){Bf[j][g][0]=v0.x;Bf[j][g][1]=v0.y;Bf[j][g][2]=v0.z;Bf[j][g][3]=v0.w;
          Bf[j][g][4]=v1.x;Bf[j][g][5]=v1.y;Bf[j][g][6]=v1.z;Bf[j][g][7]=v1.w;}
          else{bcache[((j*8+(g-4)*2)*NCW+warp)*32+lane]=v0;
               bcache[((j*8+(g-4)*2+1)*NCW+warp)*32+lane]=v1;}''')
  sub('        uint32_t o[16];','''        uint4 cb0,cb1;
        if(g>=4){cb0=bcache[((g-4)*2*NCW+warp)*32+lane];cb1=bcache[(((g-4)*2+1)*NCW+warp)*32+lane];}
        uint32_t o[16];''')
  target='for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);'
  sub(target,'''for (int j = 0; j < NT; ++j){uint32_t b0,b1;
            if(g<4){b0=Bf[j][g][2*s];b1=Bf[j][g][2*s+1];}
            else{b0=s==0?cb0.x:s==1?cb0.z:s==2?cb1.x:cb1.z;b1=s==0?cb0.y:s==1?cb0.w:s==2?cb1.y:cb1.w;}
            mma16816(acc[j],a0,a1,a2,a3,b0,b1);}''')
  sub('sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs','sizeof(Route) + ((npairs+15)&~15) + 32768')
 # Close the route-only scratch phase before any activation-cache or reduction writes.
 sub('  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1]', '  bar_consumers();\n  if constexpr (TRACE) {\n    if (tid == 0) {\n      probe[1]')
 if ws:
  sub('constexpr int NTHR = NCT;','constexpr int NTHR = NCT+128;')
  sub('__launch_bounds__(NTHR, NT == 1 ? 2 : 1)','__launch_bounds__(NTHR, 2)')
  # Consumer route loops retain the old256/128 participant count.
  sub('i += NTHR','i += NCT')
  # Producer threads do not enter consumer barriers. All threads meet once after routes are ready.
  cu=cu.replace('__syncthreads();','bar_consumers();')
  anchor='  if (tid == 0) {\n    for (int s = 0; s < nstage;'
  producer='''  if(tid>=NCT){
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;":::"memory");
    __syncthreads();
    if(tid==NCT){const int u1=rt->us1[NLOC],u2=rt->us2[NLOC];
      const int aa1=(int)((long long)u1*c/G),bb1=(int)((long long)u1*(c+1)/G);
      const int aa2=(int)((long long)u2*c/G),bb2=(int)((long long)u2*(c+1)/G);
      const int first=bb1-aa1,total=first+bb2-aa2;
      for(int i=0;i<total;++i){prefetch_one(rt,total,first,aa2,nstage,stages,fullb,emptyb,w13p,w2p);PREFETCH_L2}}
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 CONSUMER_REGS;":::"memory");
'''.replace('CONSUMER_REGS','192' if four else '104')
  if kind=='WS8L2':
   l2='''if(rt->pre.index<total){const auto&p=rt->pre;const uint8_t*src=(p.pass?w2p:w13p)+(static_cast<size_t>(p.e)*(p.pass?NT2:NT1)+p.m)*TILE_BYTES;asm volatile("cp.async.bulk.prefetch.L2.global [%0],%1;"::"l"(src),"r"(TILE_BYTES):"memory");}'''
  else:l2=''
  producer=producer.replace('PREFETCH_L2',l2)
  sub(anchor,producer+anchor)
  # Initial and steady-state copies belong to the separate producer; routing still initializes its cursor.
  a='''        for (int i = 0; i < nstage; ++i)
          prefetch_slot(rt, total_count, first_count, a1, a2, nstage, stages, fullb, emptyb, w13p, w2p);''';sub(a,'')
  a='''      for (int i = 0; i < nstage; ++i)
        prefetch_one(rt, total_count, first_count, a2, nstage, stages, fullb, emptyb, w13p, w2p);''';sub(a,'')
  a=cu.index('    if (tid == (VEC?224:0)) {');b=cu.index('    if constexpr(TRACE)if(tid==0)et=tg();',a);cu=cu[:a]+cu[b:]
  sub('    bar_consumers();\n    // deterministic token lists', '    __syncthreads(); // prefix ready:producer starts while consumers sort token lists\n    // deterministic token lists')
 # Restrict the role kernel to its admitted NT1flags69 and325;M1 stays the original S15 function.
 sub('#define VAR_FLAGS(X) X(69) X(95) X(325)','#define VAR_FLAGS(X) X(69) X(325)')
 #Only the admitted NT1 path is built;M1 andlarger M dispatch to their original kernels.
 cu=cu.replace('INFO(69) INFO(95) INFO(325)','INFO(69) INFO(325)')
 cu=cu.replace('  SET_ATTR(0)  // resident_blocks() sizes NT=1 grids with moe_var<1, 0> (same smem/threads as every NT=1 word)\n','')
 a=cu.index('  if (nt == 2) C10_CUDA_CHECK',cu.index('int64_t resident_blocks')) if '  if (nt == 2) C10_CUDA_CHECK' in cu else -1
 if a>=0:
  b=cu.index('  return n;',a);cu=cu[:a]+'  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,moe_var<1,69>,NTHR,smem_bytes(1,nstage,M)));\n'+cu[b:]
 cu=instrument(cu,ws=ws)
 assert repr(old) in cell;cell=cell.replace(repr(old),repr(cu),1);cell=cell.replace('SMALLM_FLAGS_BUILT = [69, 95, 325]','SMALLM_FLAGS_BUILT = [69, 325]').replace('(grid,64)','(grid,256)').replace('[grid,64]','[grid,256]');ast.parse(cell)
 return cell,cu
if __name__=='__main__':
 cells={};manifest={};out=R/'runs/screen/p3a_overlap';base,_,_=parent.review.original();b=base.index('def _s4_prepare_banks(')
 for kind in KINDS:
  for block in (False,True):
   cell,cu=source(kind,block);name=kind+str(int(block));tag='rvc_ov_'+name.lower();cell=parent.rename(cell,tag);cp=D/(name.lower()+'_cell.py');cp.write_text(cell);(out/(name.lower()+'.cu')).write_text(cu);cells[name]=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),prefix=tag,kind=kind,block=block,threads=256 if kind in ('CACHE','WS4') else 384)
  specs=[cells[kind+str(i)] for i in (0,1)];tag='rvc_ov_'+kind.lower();selector=f'\ndef {tag}_module():\n'+''.join(f'    {s["prefix"]}_module()\n' for s in specs)
  selector+=f'def {tag}_bind(state,flags_by_m):\n'+''.join(f'    {s["prefix"]}_bind(state,{{8:69}})\n' for s in specs)+'    return state\n'
  selector+=f"def {tag}_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n    if m==1:return _ov_original_experts(data,state,flags,stages,grid)\n    fn={specs[0]['prefix']}_experts if m<=8 else {specs[1]['prefix']}_experts\n    return fn(data,state,flags,2,grid)\n"
  wrapper=f'\ndef smallm_bind(state,flags_by_m):\n    _ov_original_bind(state,flags_by_m)\n    return {tag}_bind(state,flags_by_m)\ndef smallm_experts(data,state,flags,stages=2,grid=264):\n    return {tag}_experts(data,state,flags,stages,grid)\n';text=base[:b]+'\n'.join((R/s['cell']).read_text() for s in specs)+'\n_ov_original_bind=smallm_bind\n_ov_original_experts=smallm_experts\n'+selector+wrapper+'\n'+base[b:];ast.parse(text);p=out/(kind.lower()+'.py');p.write_text(text);pp=D/(kind.lower()+'.patch');pp.write_text(''.join(difflib.unified_diff(base.splitlines(True),text.splitlines(True),fromfile='a/'+str(parent.review.BASE.relative_to(R)),tofile='b/'+str(parent.review.BASE.relative_to(R)))));(D/(kind.lower()+'_selector.py.inc')).write_text(selector);manifest[kind]=dict(base_sha256=parent.review.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=sha(text),patch=str(pp.relative_to(R)),patch_sha256=sha(pp.read_text()),prefix=tag,depth=2,kind=kind)
 for block in (False,True):
  cell,cu=parent.parent(block);old=cu;cu=instrument(cu,False);cell=cell.replace(repr(old),repr(cu),1).replace('(grid,64)','(grid,256)').replace('[grid,64]','[grid,256]');name='CLOCK'+str(int(block));tag='rvc_ov_clock'+str(int(block));cell=parent.rename(cell,tag);cp=D/(name.lower()+'_cell.py');cp.write_text(cell);(out/(name.lower()+'.cu')).write_text(cu);cells[name]=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),prefix=tag,kind='CLOCK',block=block,threads=256)
 (D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built',list(manifest))
