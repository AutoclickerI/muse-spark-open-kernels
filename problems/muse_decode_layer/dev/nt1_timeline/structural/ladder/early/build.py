import ast,difflib,hashlib,importlib.util,json
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[6];O=R/'runs/screen/p3a_early';O.mkdir(parents=True,exist_ok=True)
p=D.parents[1]/'intpipe/build.py';sp=importlib.util.spec_from_file_location('ip',p);ip=importlib.util.module_from_spec(sp);sp.loader.exec_module(ip);base=ip.original();sha=ip.sha

def early(cu):
 a=cu.index('    if (tid == (VEC?224:0)) {');b=cu.index('    if constexpr(TRACE)if(tid==0)et=tg();',a);producer=cu[a:b];assert 'prefetch_one' in producer and 'prefetch_slot' in producer
 cu=cu[:a]+cu[b:]
 old='    __syncwarp();\n    if (lane == 0) mbar_arrive(&emptyb[st]);';assert cu.count(old)==1;cu=cu.replace(old,'    // Empty arrival moved to the last raw read;st/phase still advance after math.')
 old='        const uint4 wv = wq[g * 32];';assert cu.count(old)==1
 new='''        uint4 wv;
        if(g==7){asm volatile("ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(wv.x),"=r"(wv.y),"=r"(wv.z),"=r"(wv.w):"r"(smem_u32(wq+g*32)):"memory");}
        else wv=wq[g*32];'''
 cu=cu.replace(old,new)
 anchor='        uint32_t o[16];';assert cu.count(anchor)==1
 cu=cu.replace(anchor,'        if(g==7){__syncwarp();if(lane==0)mbar_arrive(&emptyb[st]);\n'+producer+'        }\n'+anchor)
 return cu

cells={};manifest={};basepath=ip.BASE
for arm in ('EARLY','EARLY_PRIVATE'):
 candidate=base;variants=[]
 for i,prefix in enumerate(ip.PREFIXES[1:],1):
  c=ip.cell(base,prefix);node=next(n for n in ast.parse(c).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));old=ast.literal_eval(node.value);cu=early(ip.transform(old,'PRIVATE') if arm=='EARLY_PRIVATE' else old);tag=('epr' if arm=='EARLY_PRIVATE' else 'ear')+str(i);text=c.replace(repr(old),repr(cu),1).replace(prefix,tag).replace(prefix.upper(),tag.upper());cp=D/(tag+'_cell.py');cp.write_text(text);(O/(tag+'.cu')).write_text(cu);cells[tag]=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(text),cuda_sha256=sha(cu),prefix=tag,arm=arm,M=[8] if i==1 else [16,32]);variants.append(cells[tag])
  original=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==prefix.upper()+'_CUDA');candidate=candidate.replace(ast.get_source_segment(base,original),prefix.upper()+'_CUDA='+repr(cu),1)
 p=O/(arm.lower()+'.py');p.write_text(candidate);pp=D/(arm.lower()+'.patch');pp.write_text(''.join(difflib.unified_diff(base.splitlines(True),candidate.splitlines(True),fromfile='a/'+str(basepath.relative_to(R)),tofile='b/'+str(basepath.relative_to(R)))))
 manifest[arm]=dict(base=str(basepath.relative_to(R)),base_sha256=ip.BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=sha(candidate),patch=str(pp.relative_to(R)),patch_sha256=sha(pp.read_text()),variants=variants)
(D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('EARLY_BUILT',list(cells),flush=True)
