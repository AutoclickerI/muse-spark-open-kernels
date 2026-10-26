"""One-change address arms on the exact frozen S19 small-M cells."""
import ast,base64,difflib,hashlib,json,lzma,re
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_intpipe';O.mkdir(exist_ok=True,parents=True)
BASE=R/'problems/muse_decode_layer/dev/s19/generated/s19.py';BASE_SHA='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a'
PREFIXES=('rvc_nfidx0','rvc_nbase1','rvc_nbase2');sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def original():
 packed=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();blob=next(n.args[0].value for n in ast.walk(ast.parse(packed)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');s=lzma.decompress(base64.b85decode(blob)).decode();assert sha(s)==BASE_SHA;return s

def cell(s,prefix):
 out=['import hashlib','import torch','from torch.utils.cpp_extension import load_inline']
 for n in ast.parse(s).body:
  name=n.name if isinstance(n,ast.FunctionDef) else getattr(n.targets[0],'id','') if isinstance(n,ast.Assign) else ''
  if name.startswith((prefix+'_',prefix.upper()+'_','_'+prefix.upper()+'_')):out.append(ast.get_source_segment(s,n))
 return '\n'.join(out)+'\n'

def transform(cu,arm):
 if arm=='BASE':return cu
 if arm=='AFFINE':
  if 'const uint4* base=' in cu:return cu
  # Keep pass/valid control in the same loop. Only hoist the full byte address.
  a=cu.index('#pragma unroll\n      for (int j = 0; j < NT; ++j) {',cu.index('auto unit_body'));b=cu.index('    }\n    if constexpr (TRACE)',a)
  if 'const uint4* base=' in cu[a:b]:return cu # M1 already has INDEX; unchanged alias,not a new result.
  fill=cu[a:b];needle='#pragma unroll\n        for (int g = 0; g < 8; ++g) {';assert fill.count(needle)==1
  fill=fill.replace(needle,'        const uint4* affine=reinterpret_cast<const uint4*>(row)+warp*64+q*2;\n'+needle)
  old='reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16)';assert old in fill;fill=fill.replace(old,'affine+g*8');fill=fill.replace('v0 = __ldcg(p); v1 = __ldcg(p + 1);','v0 = addr_load<true,g*128>(affine); v1 = addr_load<true,g*128+16>(affine);').replace('v0 = __ldg(p); v1 = __ldg(p + 1);','v0 = addr_load<false,g*128>(affine); v1 = addr_load<false,g*128+16>(affine);')
  begin=fill.index('        for (int g = 0; g < 8; ++g) {');brace=fill.index('{',begin);depth=1;end=brace+1
  while depth:
   depth+=(fill[end]=='{')-(fill[end]=='}');end+=1
  body=fill[brace+1:end-1];fill=fill[:begin]+''.join('{'+re.sub(r'\bg\b',str(k),body)+'}\n' for k in range(8))+fill[end:]
  cu=cu[:a]+fill+cu[b:]
  helper='template<bool CG,int OFF>__device__ __forceinline__ uint4 addr_load(const uint4* p){uint4 v;\n if constexpr(CG)asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3},[%4+%5];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p),"n"(OFF):"memory");\n else asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4+%5];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p),"n"(OFF):"memory");return v;}\n'
  cu=cu.replace('// 16 E2M1 codes',helper+'// 16 E2M1 codes')
 elif arm=='PRIVATE':
  # Initial fill stays with warp0. Transfer cursor once after its last route barrier;
  # warp7 owns it throughout the steady loop. No extra CTA/warp or stage.
  for helper in ('prefetch_one','prefetch_slot'):
   a=cu.index('__device__ __forceinline__ void '+helper+'(');b=cu.index('\n}',a)+2;part=cu[a:b]
   assert 'const uint8_t* w13p, const uint8_t* w2p)' in part
   part=part.replace('const uint8_t* w13p, const uint8_t* w2p)','const uint8_t* w13p, const uint8_t* w2p,PrefetchState* owned=nullptr)').replace('auto& p = rt->pre;','auto& p = owned?*owned:rt->pre;')
   cu=cu[:a]+part+cu[b:]
  needle='  const int q = lane & 3;';assert cu.count(needle)==1
  cu=cu.replace(needle,'  PrefetchState owned_pre;\n  if(tid==(VEC?224:0))owned_pre=rt->pre;\n'+needle)
  a=cu.index('    if (tid == (VEC?224:0)) {');b=cu.index('    if constexpr(TRACE)if(tid==0)et=tg();',a);part=cu[a:b];assert part.count('w13p, w2p);')==2;part=part.replace('w13p, w2p);','w13p, w2p,&owned_pre);');cu=cu[:a]+part+cu[b:]
 else:raise ValueError(arm)
 return cu

def main():
 s=original();nodes=ast.parse(s).body;cells={};manifest={}
 for arm in ('BASE','AFFINE','PRIVATE'):
  cand=s;variants=[]
  for i,prefix in enumerate(PREFIXES):
   name=prefix.upper()+'_CUDA';node=next(n for n in nodes if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name);old=ast.literal_eval(node.value);cu=transform(old,arm)
   cand=cand.replace(ast.get_source_segment(s,node),name+' = '+repr(cu),1)
   c=cell(s,prefix).replace(repr(old),repr(cu),1);tag='ipc_'+arm.lower()+str(i);c=c.replace(prefix,tag).replace(prefix.upper(),tag.upper());cp=D/(arm.lower()+str(i)+'_cell.py');cp.write_text(c);(O/(arm.lower()+str(i)+'.cu')).write_text(cu)
   spec=dict(cell=str(cp.relative_to(R)),cell_sha256=sha(c),cuda_sha256=sha(cu),prefix=tag,arm=arm,M=[1] if i==0 else [8] if i==1 else [16,32],unchanged=cu==old);cells[arm+str(i)]=spec;variants.append(spec)
  ast.parse(cand);p=O/(arm.lower()+'.py');p.write_text(cand)
  patch=''.join(difflib.unified_diff(s.splitlines(True),cand.splitlines(True),fromfile='a/'+str(BASE.relative_to(R)),tofile='b/'+str(BASE.relative_to(R))));pp=D/(arm.lower()+'.patch');pp.write_text(patch)
  manifest[arm]=dict(base_sha256=BASE_SHA,base=str(BASE.relative_to(R)),candidate=str(p.relative_to(R)),candidate_sha256=sha(cand),patch=str(pp.relative_to(R)),patch_sha256=sha(patch),variants=variants)
 (D/'cells.json').write_text(json.dumps(cells,indent=2)+'\n');(D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built',list(manifest))
if __name__=='__main__':main()
