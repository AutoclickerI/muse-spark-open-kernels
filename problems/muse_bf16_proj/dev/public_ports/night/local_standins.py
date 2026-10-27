"""Preserve raw-pointer/cluster launch ABI; substitute only Hopper device math."""
import importlib.util,sys
from pathlib import Path
import torch,triton,triton.language as tl
H=Path(__file__).resolve().parent;R=H.parents[4]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
prior=load('ng_prior_local',H.parent/'screen.py');prior.s19=load('ng_s21_local',R/'problems/muse_decode_layer/dev/s21/screen.py');calls=[];codec=[]
@triton.jit
def _pointer_mm(XP,WP,OP,T:tl.constexpr,N:tl.constexpr,K:tl.constexpr):
 X=tl.cast(XP,tl.pointer_type(tl.bfloat16));W=tl.cast(WP,tl.pointer_type(tl.bfloat16));O=tl.cast(OP,tl.pointer_type(tl.float32));r=tl.program_id(0)*32+tl.arange(0,32);c=tl.program_id(1)*32+tl.arange(0,32);kk=tl.arange(0,64);acc=tl.zeros((32,32),tl.float32)
 for i in range(triton.cdiv(K,64)):
  k=i*64+kk;a=tl.load(X+r[:,None]*K+k[None,:],(r[:,None]<T)&(k[None,:]<K),0);b=tl.load(W+c[:,None]*K+k[None,:],(c[:,None]<N)&(k[None,:]<K),0);acc=tl.dot(a,b.T,acc)
 tl.store(O+r[:,None]*N+c[None,:],acc,(r[:,None]<T)&(c[None,:]<N))

def verify_stream(ns,m,bank):
 seen=set()
 for (t,role),p in bank['plans'].items():
  cfg=p['cfg'];key=role,cfg
  if key in seen:continue
  seen.add(key);pk,tables,_=m._nf_group_pack(ns,bank['roles'][role],role,'v18');wr,wk,ku,s=cfg;assert wr==1;rr,cc,bb=pk['data'].shape;g=rr*s*wk;gw=torch.arange(g,device='cuda');row=(gw//wk)%rr;by=(gw//wk)//rr;chunk=(by*wk+gw%wk)[None,:]*ku+torch.arange(ku,device='cuda')[:,None];restored=torch.empty_like(pk['data']);restored[row[None,:],chunk]=p['wp'].view(ku,g,bb);assert torch.equal(restored,pk['data']),(role,'stream inverse')
  assert torch.equal(p['esc'][:,32:].long(),tables[row]),(role,'per-row origins')
  entries=p['esc'][:,:32];mask=entries!=-1;indices=torch.arange(g,device='cuda')[:,None].expand(g,32)[mask];e=entries[mask].long()&0xffffffff;er=row[indices];ec=(by[indices]*wk+indices%wk)*ku+(e>>18);el=(e>>13)&31;reg=(e>>8)&31;ev=(reg%16)*2+((e>>7)&1);xr=e&127;half=reg//16
  def canonical(parts):
   r,c,l,v,x,h=[p.long() for p in parts];return (((((r*cc+c)*32+l)*32+v)*128+x)*2+h).sort().values
  assert torch.equal(canonical((er,ec,el,ev,xr,half)),canonical(pk['esc'])),(role,'escape inverse')
  codec.append(dict(role=role,cfg=cfg,weights=bank['roles'][role].numel(),escaped=mask.sum().item(),bytes=p['wp'].numel()+p['esc'].numel()*4))

def install(m,forced=True):
 prior.install(m);lookup={};original_vendor=m._nf_vendor
 def vendor():
  ns=original_vendor()
  def build(insts,label):
   if not label.startswith('p10_night_dsm_'):return original_build(insts,label)
   class Module:
    def v18s_launch(self,*a):
     idx,wp,x,out,esc,t,n,k,wr,wk,ku,s,tab,u,stream,query=a;inst=insts[idx];assert (t//8,n,k,wr,wk,ku,s)==(inst[1],*inst[8:14]);assert inst[4]==(3 if s>1 else 0)
     calls.append(('NF',t,n,k,s,wk,inst[3],query))
     if query:return 0
     w=lookup[wp];_pointer_mm[(triton.cdiv(t,32),triton.cdiv(n,32))](x,w.data_ptr(),out,t,n,k,num_warps=4,num_stages=2);return 0
   return Module()
  original_build=ns['v18s_build'];ns['v18s_build']=build;return ns
 m._nf_vendor=vendor;old_prepare=m.ng_prepare
 def prepare(ns,st):
  bank=old_prepare(ns,st)
  for (t,role),p in bank['plans'].items():lookup[p['wp'].data_ptr()]=bank['roles'][role]
  lookup[bank['tma_w'].data_ptr()]=bank['roles']['pool_router']
  recovered=bank['tma_w'].reshape(72,64,64,128).permute(0,2,1,3).reshape(4608,8192);assert torch.equal(recovered.view(torch.int16),bank['roles']['pool_router'].view(torch.int16))
  v=vendor();exec(m.PP_GPU_PACK_SOURCE,v);verify_stream(v,m,bank);return bank
 m.ng_prepare=prepare
 class Cluster:
  def warmup(self,*a,**kw):
   assert kw['launch_pdl'] is True
   class Run:
    launch_pdl=True
    def __call__(self,gx,gy,gz,stream,function,pm,metadata,h1,h2,*args):
     xd,wd,out,t,n,k,bm,bn,bk,s,slot=args;assert (gx,gy,gz,pm[1],s,slot)==(1,72,1,3,3,16);calls.append(('TMA_cluster',t,n,k,s,bk,kw['num_stages'],self.launch_pdl));prior.portable.mm(xd.base,lookup[wd.base.data_ptr()],out)
   class Compiled:
    packed_metadata=(4,1,65536);function=None;run=Run()
    def _init_handles(self):pass
    def launch_metadata(self,*a):return None
   return Compiled()
 m._ng_cluster_pool=Cluster()
 if not hasattr(m,'pp_pool_project'):return  # composition without a parent pool schedule (S22 handoff)
 old_pool=m.pp_pool_project;m._pp_pool_tma_early=m._cbf_skinny_gemm_tma_tiled_ppdl
 def pool_project(x,bank,pool,name):
  for (role,bk),wp in bank['tiled'].items():m._pp_local_lookup[wp.data_ptr(),bank['roles'][role].shape[0]]=bank['roles'][role]
  return old_pool(x,bank,pool,name)
 m.pp_pool_project=pool_project
