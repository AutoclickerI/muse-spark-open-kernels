"""Independent S15 small-M dependency, norm identity, sharding and exact scale-packet arms."""
import ast,difflib,hashlib,json,sys
from pathlib import Path
D=Path(__file__).resolve().parent;P=D.parent;R=D.parents[4]
sys.path.insert(0,str(P))
from build_epilogue_pair import source as pair_source
BASE=R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
BASE_SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'
PREFIX={'BLOCK':'rvb','CANCEL':'rvc','PAIR':'rvp','SHARD2':'rvs2','SHARD4':'rvs4','SHARD8':'rvs8','PACK5':'rvq'}

def original():
 text=BASE.read_text();assert hashlib.sha256(text.encode()).hexdigest()==BASE_SHA
 a=text.index('SMALLM_VAR_CUDA = ');b=text.index('def _s4_prepare_banks(',a)
 cell=text[a:b];cuda=next(ast.literal_eval(n.value) for n in ast.parse(cell).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='SMALLM_VAR_CUDA')
 return text,cell,cuda

def source(arm):
 base,cell,cuda=original()
 if arm=='PAIR' or arm.startswith('SHARD'):
  cell,cuda=pair_source();cell=cell.replace('epit_','smallm_').replace('EPIT_','SMALLM_VAR_').replace('_SMALLM_VAR_MODULE','_SMALLM_MODULE').replace('SMALLM_VAR_FLAGS_BUILT','SMALLM_FLAGS_BUILT')
  # Pair's module/cell is the previously H200-gated implementation, rebased onto identical S15 math.
 old=cuda
 def change(before,after,count=None):
  nonlocal cuda
  n=cuda.count(before);assert n and (count is None or n==count),(arm,before,n,count)
  cuda=cuda.replace(before,after)
 if arm=='BLOCK':
  change('enum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };','enum { C_DONE = 0, C_ZERO = 128, C_BAR = 129, C_GEN = 130 };',1)
  change('cnt + C_DONE + cur_e, seg_tiles','cnt + C_DONE + cur_e*4+cur_nb, seg_tiles',2)
  change('const int target = NT1 * (SLOT ? 1 : rt->nblk[e]);','const int target = NT1;',1)
  change('cnt + C_DONE + e) < target','cnt + C_DONE + e*4+nb) < target',1)
  change('i < NLOC; ++i) cnt[C_DONE + i] = 0','i < NLOC*4; ++i) cnt[C_DONE + i] = 0',2)
  cell=cell.replace('torch.zeros(64,','torch.zeros(160,')
 elif arm=='CANCEL':
  helper=r'''
__device__ __forceinline__ float cancel_norm(const float* meta,int e,float sum){
 const float eps=meta[32+e],sgn=meta[64+e],mean=sum*(1.f/HD);
 if(sgn==0.f||mean==0.f)return 0.f;
 if(eps>0.f&&isfinite(eps))return sgn*rsqrtf(mean+eps);
 const double a=(double)meta[e]*256.0;
 return (float)((a<0.0?-1.0:1.0)/sqrt((double)mean+1e-8/(a*a)));
}
'''
  change('struct PrefetchState {',helper+'\nstruct PrefetchState {',1)
  change('      const float sc = gs2[e] * 256.f;','',1)
  change('        v *= sc;','',1)
  change('rsqrtf(__ldcg(ss + posl) * (1.f / HD) + 1e-8f)','cancel_norm(gs2,SLOT?rt->erank[posl]:(int)eflat[pair_of[posl]],__ldcg(ss+posl))',1)
  change('rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f)','cancel_norm(gs2,SLOT?rt->erank[pos]:(int)eflat[pair_of[pos]],__ldcg(ss+pos))',1)
  anchor='    return state\n\n\ndef smallm_experts'
  assert anchor in cell
  cell=cell.replace(anchor,"    a=state['gs2'].double()*256.0\n    state['cancel_meta']=torch.cat((state['gs2'],(1e-8/(a*a)).float(),a.sign().float())).contiguous()\n"+anchor)
  cell=cell.replace("state['gs13'], state['gs2'], inter", "state['gs13'], state['cancel_meta'], inter")
 elif arm.startswith('SHARD'):
  k=int(arm[5:]);change('constexpr int HD = 4096;',f'constexpr int SHARDS={k};\nconstexpr int HD = 4096;',1)
  change('ss[off+idx]=0.f;', 'for(int sh=0;sh<SHARDS;++sh)ss[(off+idx)*SHARDS+sh]=0.f;',1)
  change('ss[off + idx] = 0.f;', 'for(int sh=0;sh<SHARDS;++sh)ss[(off+idx)*SHARDS+sh]=0.f;',1)
  change('atomicAdd(ss+off+n,s);','atomicAdd(ss+(off+n)*SHARDS+(c&(SHARDS-1)),s);',1)
  change('atomicAdd(ss + npos, t2);','atomicAdd(ss+npos*SHARDS+(c&(SHARDS-1)),t2);',1)
  change('atomicAdd(ss + off + idx, t2);','atomicAdd(ss+(off+idx)*SHARDS+(c&(SHARDS-1)),t2);',1)
  change('ss[i] = 0.f;', 'for(int sh=0;sh<SHARDS;++sh)ss[i*SHARDS+sh]=0.f;',1)
  helper='''\n__device__ __forceinline__ float shard_sum(const float* ss,int pos){float v=0.f;
#pragma unroll
for(int sh=0;sh<SHARDS;++sh)v+=__ldcg(ss+pos*SHARDS+sh);return v;}\n'''
  change('struct PrefetchState {',helper+'\nstruct PrefetchState {',1)
  change('__ldcg(ss + posl)','shard_sum(ss,posl)',1);change('__ldcg(ss + pos)','shard_sum(ss,pos)',1)
  cell=cell.replace('ss = torch.empty((p,),','ss = torch.empty((p * '+str(k)+',),')
 elif arm=='PACK5':
  cuda=pack_cuda(cuda)
  cell=cell.replace("    return state\n\n\ndef smallm_experts", "    state['banks']['packet5']={k:pack5_bank(v) for k,v in state['banks']['lop3'].items()}\n    return state\n\n\ndef smallm_experts")
  cell=cell.replace("bank = state['banks']['lop3']", "bank = state['banks']['packet5']")
 elif arm!='PAIR':raise ValueError(arm)
 assert repr(old) in cell,(arm,'CUDA literal not present')
 cell=cell.replace(repr(old),repr(cuda),1)
 if arm=='PACK5':cell+='\n'+(D/'pack5_python.py.inc').read_text()
 tag=PREFIX[arm];cell=cell.replace('smallm_',tag+'_').replace('_SMALLM_MODULE','_'+tag.upper()+'_MODULE').replace('SMALLM_FLAGS_BUILT',tag.upper()+'_FLAGS_BUILT').replace('SMALLM_VAR_',tag.upper()+'_')
 ast.parse(cell)
 return cell,cuda

def pack_cuda(cuda):
 helper=(D/'pack5_cuda.py.inc').read_text()
 cuda=cuda.replace('constexpr int TILE_BYTES = 36864;','constexpr int TILE_BYTES = 36896;')
 cuda=cuda.replace('struct PrefetchState {',helper+'\nstruct PrefetchState {')
 old='const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;'
 assert old in cuda;cuda=cuda.replace(old,'int packet_bytes;const uint8_t* src=packet_tile(base,static_cast<size_t>(e)*ntile+m,packet_bytes);')
 old='const uint8_t* src = (pass ? w2p : w13p) + (static_cast<size_t>(e) * (pass ? NT2 : NT1) + m) * TILE_BYTES;'
 assert old in cuda;cuda=cuda.replace(old,'int packet_bytes;const uint8_t* src=packet_tile(pass?w2p:w13p,static_cast<size_t>(e)*(pass?NT2:NT1)+m,packet_bytes);')
 cuda=cuda.replace('mbar_expect_tx(&fullb[st], TILE_BYTES);','mbar_expect_tx(&fullb[st], packet_bytes);')
 cuda=cuda.replace('src, TILE_BYTES, &fullb[st], pol','src, packet_bytes, &fullb[st], pol')
 cuda=cuda.replace('src, TILE_BYTES, &fullb[st], p.pol_first','src, packet_bytes, &fullb[st], p.pol_first')
 old='''      const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);
      const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};'''
 new='''      uint32_t scw[4];
#pragma unroll
      for(int k=0;k<4;++k){uint32_t sw=0;
#pragma unroll
        for(int j=0;j<4;++j)sw|=packet_scale(tile,(warp*32+lane)*16+k*4+j)<<(j*8);
        scw[k]=sw;}'''
 assert old in cuda;cuda=cuda.replace(old,new)
 return cuda

def build(arms):
 base,_,_=original();a=base.index('SMALLM_VAR_CUDA = ');b=base.index('def _s4_prepare_banks(',a);manifest={}
 out=R/'runs/screen/p3a_review';out.mkdir(parents=True,exist_ok=True)
 for arm in arms:
  cell,cuda=source(arm);tag=PREFIX[arm]
  # Keep the original dispatch and M1 path; the new block dependency applies only at16/32.
  wrapper=f'''\n_review_original_bind=smallm_bind
_review_original_experts=smallm_experts
def smallm_bind(state,flags_by_m):
    _review_original_bind(state,flags_by_m)
    {tag}_bind(state,flags_by_m)
    return state
def smallm_experts(data,state,flags,stages=2,grid=264):
    m=data['hidden_states'].shape[0]
    if {'m in (16,32)' if arm=='BLOCK' else 'm<=32'}:return {tag}_experts(data,state,flags,stages,grid)
    return _review_original_experts(data,state,flags,stages,grid)
'''
  cand=base[:b]+cell+wrapper+'\n'+base[b:];ast.parse(cand)
  p=out/(arm.lower()+'.py');p.write_text(cand)
  cp=D/(arm.lower()+'_cell.py');cp.write_text(cell)
  rel=str(BASE.relative_to(R));patch=''.join(difflib.unified_diff(base.splitlines(True),cand.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel));pp=D/(arm.lower()+'.patch');pp.write_text(patch)
  cu=out/(arm.lower()+'.cu');cu.write_text(cuda)
  manifest[arm]=dict(base_sha256=BASE_SHA,candidate=str(p.relative_to(R)),candidate_sha256=hashlib.sha256(cand.encode()).hexdigest(),cell=str(cp.relative_to(R)),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),patch=str(pp.relative_to(R)),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),prefix=tag,parent='PAIR' if arm.startswith('SHARD') else 'S15',one_change=arm)
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('arms',nargs='*',default=list(PREFIX));a=p.parse_args();build(a.arms)
