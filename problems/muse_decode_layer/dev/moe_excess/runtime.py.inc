from concurrent.futures import ThreadPoolExecutor
NAMES=('BASE','AA','MATCH','SHADOW','L0','L1','L2','L3','MATH','INIT','STREAM','HIST')
MODES={'BASE':-2,'AA':-2,'MATCH':-1,'SHADOW':4,'L0':0,'L1':1,'L2':2,'L3':3,'MATH':5,'INIT':6,'STREAM':7,'HIST':4}
NVML=None;TELEMETRY_ERROR=None

def telemetry():
 global NVML,TELEMETRY_ERROR
 import ctypes
 try:
  if NVML is None:
   lib=ctypes.CDLL('libnvidia-ml.so.1');rc=lib.nvmlInit_v2();assert rc==0,rc;handle=ctypes.c_void_p();rc=lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0),ctypes.byref(handle));assert rc==0,rc;NVML=lib,handle
  lib,h=NVML
  def read(fn,*args):
   x=ctypes.c_uint();rc=getattr(lib,fn)(h,*args,ctypes.byref(x));assert rc==0,(fn,rc);return x.value
  return [round(time.monotonic()*1000),read('nvmlDeviceGetClockInfo',ctypes.c_uint(1)),read('nvmlDeviceGetClockInfo',ctypes.c_uint(2)),read('nvmlDeviceGetTemperature',ctypes.c_uint(0)),read('nvmlDeviceGetPowerUsage')]
 except Exception as e:TELEMETRY_ERROR=repr(e);return [round(time.monotonic()*1000),-1,-1,-1,-1]

def extension(pm=False):
 from torch.utils.cpp_extension import load_inline
 cu=B['pmcuda' if pm else 'cuda'];arch='90a' if torch.cuda.get_device_capability()==(9,0) else ''.join(map(str,torch.cuda.get_device_capability()));tag='moe_excess_'+sha(cu)[:12]+'_'+arch
 cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nstd::vector<int64_t> xl_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool);'
 return load_inline(name=tag,cpp_sources=cpp,cuda_sources=cu,functions=['xl_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def prepare_moe(base,w):
 dev=w['w13'].device;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 raw=dict(gs13=w['w13_gs'],gs2=w['w2_gs'],pnw=w['post_norm_w'],grid=sms,cnt=torch.zeros(base.NCNT,dtype=torch.int32,device=dev))
 native=base._prepare_recoded_bank(w,raw,None);heavy=base.nf_perm(base.nf_wide(base.issue_prepare(base.nf_ilv(w))));base.NF_OPTS['pmz']=base.p3b_module().pmz_init();base.p3b_module().ch_info(4,0);base.p3b_module().ch_info(4,1)
 return dict(native=native,heavy=heavy,streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()))

def fixture(layer,w,M,seed):
 # Same upstream layer weight/router and stream/draft statistics, with a
 # synthetic pre-pool activation. No attention-front timing is attributed here.
 dev=w['router'].device;gen=torch.Generator(device=dev).manual_seed(seed);mean,shared=layer._stream_basis(str(dev));coef=torch.randn((M,4),generator=gen,device=dev);scale=torch.ones((M//8,8,1),device=dev);scale[:,1:]=layer.DRAFT_SHARE;scale=scale.reshape(M,1)
 stream=scale*(layer.STREAM_MEAN*mean+(layer.STREAM_SHARED/4)**.5*(coef@shared))+(1-layer.STREAM_SHARED*scale.square()).sqrt()*torch.randn((M,8192),generator=gen,device=dev)
 h=layer._rms(stream)*(1+w['pre_ffn_norm'].float());pooled=h@w['pooler'].float().T;logits=h@w['router'].float().T;scores=torch.sigmoid(logits*layer.ROUTER_MULTIPLIER);ids=(scores+w['router_bias'].float()).topk(8,-1).indices;start=torch.bincount((ids//32).flatten(),minlength=8).argmax()*32;lids=(ids-start)%256;lids=torch.where(lids<32,lids,-1).int();tw=scores.gather(1,ids);tw=(tw/(tw.sum(-1,keepdim=True)+1e-15)).contiguous();x=(layer._rms(pooled)*(1+w['pre_expert_norm'].float())).bfloat16()
 return dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096,bs=M,top_k=8))

def make_plan(base,d,s):
 p=s19_plan(base,d,s);p['steps'][0][1]();torch.cuda.synchronize();return p

def cut(base,d,s,ext,pm,p,phase,arm):
 # Metadata and current-fixture ACT are initialized once and then frozen for
 # every arm, because S19 route's atomic slot order can change across replays.
 M=d['hidden_states'].shape[0];h13,h2,nv,ids,tw,pos,tasks,l13,l2=p['owners'];stage=phase=='w13';hv=h13 if stage else h2;lv=l13 if stage else l2
 ptr=lambda ts:[t.data_ptr() for t in ts];vectors=[ptr(hv),ptr(lv),ptr(nv)]
 sms=torch.cuda.get_device_properties(d['hidden_states'].device).multi_processor_count;nstage=2
 grids=[sms*4,sms*4,sms*base.p3b_module().pmz_resident(nstage,M,int(stage))]
 traces=[torch.zeros((g,8),dtype=torch.int64,device=d['hidden_states'].device) for g in grids]
 mode=MODES[arm];info=[]
 for family in range(3):
  mod=pm if family==2 and not stage and mode!=7 else ext
  v=mod.xl_run(vectors[family],family,int(stage),mode,M,nstage,grids[family],0,traces[family].data_ptr(),False,arm=='HIST');info.append(list(v))
 side,fork,join,light,ljoin=s['streams']
 def one(family):
  mod=pm if family==2 and not stage and mode!=7 else ext
  mod.xl_run(vectors[family],family,int(stage),mode,M,nstage,grids[family],torch.cuda.current_stream().cuda_stream,traces[family].data_ptr(),True,arm=='HIST')
 def call():
  parent=torch.cuda.current_stream();fork.record(parent);side.wait_event(fork);light.wait_event(fork)
  with torch.cuda.stream(side):one(0);join.record()
  with torch.cuda.stream(light):one(1);ljoin.record()
  one(2);parent.wait_event(join);parent.wait_event(ljoin)
 return dict(fn=call,trace=traces,resources=info,grids=grids,owners=p)

def histograms(traces,seed0):
 import collections
 rows=[t.cpu().tolist() for t in traces];origin=min(v[0] for rr in rows for v in rr);answer=[];allsm={}
 def quant(x):return [round(x[min(len(x)-1,round((len(x)-1)*q))]/256) for q in (0,.5,1)] if x else [0]*3
 for bi,rr in enumerate(rows):
  joint=collections.Counter()
  for entry,end,sm,tiles,mask,tokens,n32,n64 in rr:
   joint[(tiles,int(mask).bit_count(),tokens,n32,n64)]+=1;z=allsm.setdefault(sm,[0,0,0,2**63-1,0]);z[0]+=1;z[1]+=tiles*(8 if bi==0 else 4 if bi==1 else 1);z[2]|=mask;z[3]=min(z[3],entry-origin);z[4]=max(z[4],end-origin)
  finishes=sorted(v[1]-origin for v in rr);entries=sorted(v[0]-origin for v in rr);active=[v for v in rr if v[3]>0];durations=sorted(v[1]-v[0] for v in active)
  cta_hist=[]
  for index in (0,1):
   c=collections.Counter()
   for k,n in joint.items():c[k[index]]+=n
   cta_hist.append([list(k) for k in sorted(c.items())])
  last=max(rr,key=lambda v:v[1]);last=[last[0],last[1],last[2],last[3],int(last[4]).bit_count(),last[5]]
  z=dict(CTA_hist_tiles_experts=cta_hist,entry_minmax256=[quant(entries)[i] for i in (0,2)],finish_median256=quant(finishes)[1],active_duration_quantile256=quant(durations),active_CTAs=len(active),last_CTA=last,tail_after_median_ns=finishes[-1]-finishes[len(finishes)//2])
  for key in ('last_CTA',):
   if z[key]:z[key][0]-=origin;z[key][1]-=origin
  answer.append(z)
 smjoint=collections.Counter((v[0],v[1],v[2].bit_count()) for v in allsm.values())
 sm_hist=[]
 for index in range(3):
  c=collections.Counter()
  for k,n in smjoint.items():c[k[index]]+=n
  sm_hist.append([list(k) for k in sorted(c.items())])
 result=dict(branches=answer,SM_hist_CTAs_weighttiles_experts=sm_hist,SM_finish_quantile256=quant(sorted(v[4] for v in allsm.values())),SMs=len(allsm),tick_ns=256)
 return result

def encode_result(result):
 # Integer arrays use exact delta/zigzag varints, with an ordered schema.
 # Timing ns remain exact. Only histogram CDF/per-SM timestamps have the
 # prospectively declared 256/512ns bins; last-CTA stamps remain exact.
 binary=bytearray()
 def vint(v):
  v=v*2 if v>=0 else -v*2-1
  while v>=128:binary.append((v&127)|128);v>>=7
  binary.append(v)
 def walk(x):
  if isinstance(x,list) and len(x)>1 and all(isinstance(r,list) and len(r)==len(x[0]) and r and all(type(v) is int for v in r) for r in x):
   for col in zip(*x):
    prev=0
    for v in col:vint(v-prev);prev=v
   return ['$matrix',len(x),len(x[0])]
  if isinstance(x,list) and x and all(type(v) is int for v in x):
   prev=0
   for v in x:vint(v-prev);prev=v
   return ['$int',len(x)]
  if isinstance(x,list):return [walk(v) for v in x]
  if isinstance(x,tuple):return walk(list(x))
  if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
  return x
 meta=json.dumps(walk(result),separators=(',',':')).encode();raw=struct.pack('<I',len(meta))+meta+binary
 # Varint bytes are weakly position-correlated. This fixed byte-context setup
 # leaves more room in the public packet than LZMA's text-oriented defaults.
 return base64.b85encode(lzma.compress(raw,filters=[dict(id=lzma.FILTER_LZMA2,preset=9,lc=1,lp=0,pb=0)])).decode()
