from concurrent.futures import ThreadPoolExecutor
# Per family (heavy, medium, small): (mode, depth, shared kind, hist); None = branch not launched.
# Modes: -2 original kernel, 0 ladder L0 (sinked), 4 full clone, 7 unique stream, 8 L0N loads only,
# 9 NOX, 10 REUSE, 11 ISSUE, 12 PF (full math + L2 prefetch), 13 PFN (L0N + L2 prefetch).
# Kinds: 0 original, 1 +2 KiB scratch, 2 force 1 CTA/SM, 3 exact ring, 4 force 2 CTAs/SM.
def _all(mode,kind=1,hist=False):return [(mode,4,kind,hist)]*3
def _heavy(cfg,rest=(8,4,1,False)):return [cfg,rest,rest]
ARMS_J={'BASE':_all(-2,0),'AA':_all(-2,0),'L0':_all(0),'L0N':_all(8),'NOX':_all(9),'REUSE':_heavy((10,4,1,False)),
 'ISSUE':_heavy((11,4,1,False)),'RES1':_heavy((8,4,2,False)),'RES2S3':_heavy((8,3,4,False)),'RES3':_heavy((8,3,3,False)),
 'PFN':_all(13),'PF':_all(12),'STREAM':_all(7),'HISTN':_all(8,1,True)}
def _one(i,cfg):return [cfg if j==i else None for j in range(3)]
ARMS_I={'BASE':_all(-2,0),'AA':_all(-2,0),'L0N':_all(8),'STREAM':_all(7)}
for i,f in enumerate('HMS'):
 ARMS_I[f+'_BASE']=_one(i,(-2,4,0,False));ARMS_I[f+'_L0N']=_one(i,(8,4,1,False));ARMS_I[f+'_ST']=_one(i,(7,4,1,False))
ARMS=ARMS_J if ARMSET=='J' else ARMS_I
NAMES=tuple(ARMS)
FULL=tuple(n for n,c in ARMS.items() if all(x is not None and x[0] in (-2,4,12) for x in c))
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
  return [read('nvmlDeviceGetClockInfo',ctypes.c_uint(1)),read('nvmlDeviceGetClockInfo',ctypes.c_uint(2)),read('nvmlDeviceGetTemperature',ctypes.c_uint(0)),read('nvmlDeviceGetPowerUsage')//1000]
 except Exception as e:TELEMETRY_ERROR=repr(e);return [-1,-1,-1,-1]

def extension(pm=False):
 from torch.utils.cpp_extension import load_inline
 cu=B['pmcuda' if pm else 'cuda'];arch='90a' if torch.cuda.get_device_capability()==(9,0) else ''.join(map(str,torch.cuda.get_device_capability()));tag='moe_split_'+sha(cu)[:12]+'_'+arch
 cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nstd::vector<int64_t> xl_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool,int64_t);'
 return load_inline(name=tag,cpp_sources=cpp,cuda_sources=cu,functions=['xl_run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def prepare_moe(base,w):
 dev=w['w13'].device;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 raw=dict(gs13=w['w13_gs'],gs2=w['w2_gs'],pnw=w['post_norm_w'],grid=sms,cnt=torch.zeros(base.NCNT,dtype=torch.int32,device=dev))
 native=base._prepare_recoded_bank(w,raw,None);heavy=base.nf_perm(base.nf_wide(base.issue_prepare(base.nf_ilv(w))));base.NF_OPTS['pmz']=base.p3b_module().pmz_init();base.p3b_module().ch_info(4,0);base.p3b_module().ch_info(4,1)
 return dict(native=native,heavy=heavy,streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()))

def fixture(layer,w,M,seed):
 # Identical to the ladder (1b75cd73): upstream layer weights/router and stream/draft statistics,
 # synthetic pre-pool activation, real FP32 router membership and normalization.
 dev=w['router'].device;gen=torch.Generator(device=dev).manual_seed(seed);mean,shared=layer._stream_basis(str(dev));coef=torch.randn((M,4),generator=gen,device=dev);scale=torch.ones((M//8,8,1),device=dev);scale[:,1:]=layer.DRAFT_SHARE;scale=scale.reshape(M,1)
 stream=scale*(layer.STREAM_MEAN*mean+(layer.STREAM_SHARED/4)**.5*(coef@shared))+(1-layer.STREAM_SHARED*scale.square()).sqrt()*torch.randn((M,8192),generator=gen,device=dev)
 h=layer._rms(stream)*(1+w['pre_ffn_norm'].float());pooled=h@w['pooler'].float().T;logits=h@w['router'].float().T;scores=torch.sigmoid(logits*layer.ROUTER_MULTIPLIER);ids=(scores+w['router_bias'].float()).topk(8,-1).indices;start=torch.bincount((ids//32).flatten(),minlength=8).argmax()*32;lids=(ids-start)%256;lids=torch.where(lids<32,lids,-1).int();tw=scores.gather(1,ids);tw=(tw/(tw.sum(-1,keepdim=True)+1e-15)).contiguous();x=(layer._rms(pooled)*(1+w['pre_expert_norm'].float())).bfloat16()
 return dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096,bs=M,top_k=8))

def make_plan(base,d,s):
 p=s19_plan(base,d,s);p['steps'][0][1]();torch.cuda.synchronize();return p

def cut(base,d,s,ext,pm,p,phase,arm):
 # Metadata and current-fixture ACT are produced once and frozen for every arm (S22 route's atomic
 # slot order can change across replays). Production phase structure: heavy on a side stream,
 # medium on a light stream, small on the parent, joined.
 M=d['hidden_states'].shape[0];h13,h2,nv,ids,tw,pos,tasks,l13,l2=p['owners'];stage=phase=='w13';hv=h13 if stage else h2;lv=l13 if stage else l2
 ptr=lambda ts:[t.data_ptr() for t in ts];vectors=[ptr(hv),ptr(lv),ptr(nv)]
 sms=torch.cuda.get_device_properties(d['hidden_states'].device).multi_processor_count;nstage=2
 grids=[sms*4,sms*4,sms*base.p3b_module().pmz_resident(nstage,M,int(stage))]
 traces=[torch.zeros((g,8),dtype=torch.int64,device=d['hidden_states'].device) for g in grids]
 cfg=ARMS[arm];info=[]
 def mod(family,c):return pm if family==2 and not stage and c[0]!=7 else ext
 for family,c in enumerate(cfg):
  if c is None:info.append(None);continue
  info.append(list(mod(family,c).xl_run(vectors[family],family,int(stage),c[0],c[1],M,nstage,grids[family],0,traces[family].data_ptr(),False,c[3],c[2])))
 side,fork,join,light,ljoin=s['streams']
 def one(family):
  c=cfg[family]
  if c is not None:mod(family,c).xl_run(vectors[family],family,int(stage),c[0],c[1],M,nstage,grids[family],torch.cuda.current_stream().cuda_stream,traces[family].data_ptr(),True,c[3],c[2])
 def call():
  parent=torch.cuda.current_stream();fork.record(parent);side.wait_event(fork);light.wait_event(fork)
  with torch.cuda.stream(side):one(0);join.record()
  with torch.cuda.stream(light):one(1);ljoin.record()
  one(2);parent.wait_event(join);parent.wait_event(ljoin)
 return dict(fn=call,trace=traces,resources=info,grids=grids,owners=p)

def histograms(traces):
 # Per launched branch, 16 ints: entry/finish/active-duration quantiles 0/.5/1 (256 ns bins, from the
 # first CTA entry of the phase), active CTAs, exact last CTA [entry, finish ns, SM, tiles, experts,
 # tokens]. Tile/expert marginals follow from the counts and grid and are not repeated.
 rows=[t.cpu().tolist() for t in traces];origin=min(v[0] for rr in rows for v in rr if v[1]);answer=[]
 def quant(x):return [round(x[min(len(x)-1,round((len(x)-1)*q))]/256) for q in (0,.5,1)] if x else [0]*3
 for rr in rows:
  rr=[v for v in rr if v[1]]
  if not rr:answer.append([0]*16);continue
  finishes=sorted(v[1]-origin for v in rr);entries=sorted(v[0]-origin for v in rr);active=[v for v in rr if v[3]>0];durations=sorted(v[1]-v[0] for v in active)
  last=max(rr,key=lambda v:v[1])
  answer.append(quant(entries)+quant(finishes)+quant(durations)+[len(active),last[0]-origin,last[1]-origin,last[2],last[3],int(last[4]).bit_count(),last[5]])
 return answer

def encode_result(result):
 # Exact delta/zigzag varints with an ordered schema (same codec as the ladder reader).
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
 return base64.b85encode(lzma.compress(raw,filters=[dict(id=lzma.FILTER_LZMA2,preset=9,lc=1,lp=0,pb=0)])).decode()
