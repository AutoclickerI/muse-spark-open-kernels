LAB_RESOURCES={'compile_s':130,'run_s':490,'output_bytes':8400,'context_poison':'none'}
SEEDS=tuple(5671243+7919*i for i in range(8))
CASES=((1,8,2048),(16,8,2048),(1,8,0),(16,8,0),(32,8,0))
def module(name,source):
 f='<p5-fixed-'+name+'>';linecache.cache[f]=(len(source),None,source.splitlines(True),f)
 m=types.ModuleType('p5_fixed_'+name);m.__file__=f;exec(compile(source,f,'exec'),m.__dict__);return m

def lab(deadline,weights):
 start=time.monotonic();fail=[];counts=dict(official=0,scored=0,stress=0,partial=0)
 ck=module('TASK',_BUNDLE['checker']).__dict__
 arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
 if arch!='90a':ck['_fa3_scheduler_metadata']=lambda *a,**kw:None
 from torch.utils.cpp_extension import load_inline
 built={}
 def load(cu,cpp,funcs):
  tag=hashlib.sha256((cu+cpp).encode()).hexdigest()[:14]
  if tag not in built:built[tag]=load_inline(name='p5_fixed_'+tag+'_'+arch,cpp_sources='#include <pybind11/stl.h>\n'+cpp,cuda_sources=cu,
   functions=funcs,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
  return built[tag]
 guard=load('#include <cuda_runtime.h>\n#include <cstdint>\n#include <vector>\n#include <stdexcept>\nint64_t graph_kernels(int64_t p){size_t n=0;auto g=(cudaGraph_t)p;auto e=cudaGraphGetNodes(g,nullptr,&n);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));std::vector<cudaGraphNode_t>x(n);cudaGraphGetNodes(g,x.data(),&n);int k=0;for(auto a:x){cudaGraphNodeType t;cudaGraphNodeGetType(a,&t);k+=t==cudaGraphNodeTypeKernel;}return k;}\n',"#include <cstdint>\nint64_t graph_kernels(int64_t);",['graph_kernels'])
 base=module('S19',_BUNDLE['source']+_BUNDLE['adapter']);base.ATTN_PORT_CUDA_SRC=_BUNDLE['ATTN_PORT_CUDA_SRC'];base.VQ8_CUDA=_BUNDLE['VQ8_CUDA']
 if install:=globals().get('ROOF_LOCAL_INSTALL'):install(base)
 qmods={'A':load(_BUNDLE['ATTN_PORT_CUDA_SRC'],_BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])}
 base._ATTN_PORT_EXT=qmods['A']
 pm=load(_BUNDLE['PARTIAL_CUDA'],_BUNDLE['PARTIAL_CPP'],['attn_forward','occupancy','max_cluster']);base._PP_MOD={False:pm}
 vmods={}
 for n in ('A','N','G','F'):
  source=_BUNDLE['VQ8_CUDA'] if n=='A' else _BUNDLE[n]
  if arch=='90a':vmods[n]=load(source,_BUNDLE['VQ8_CPP'],['vq8_forward','vq8_resources'])
  else:vmods[n]=globals()['CHAIN_LOCAL'].native(base,source)
 base._VQ8_MOD=vmods['A'];base._vq8_module=lambda:vmods['A']
 alt=module('S19_SCHEDULE',_BUNDLE['source']+_BUNDLE['sched']+_BUNDLE['adapter'])
 if install:install(alt)
 alt._ATTN_PORT_EXT=qmods['A'];alt._PP_MOD={False:pm};alt._VQ8_MOD=vmods['A'];alt._vq8_module=lambda:vmods['A']
 sched=alt.__dict__;sched['_p5b_sched_init']('cuda')
 bridge=load_inline(name='p5b_fa_bridge_'+hashlib.sha256(_BUNDLE['bridge'].encode()).hexdigest()[:14],cpp_sources=_BUNDLE['bridge'],functions=['fa_bind','fa_probe','fa_main'],extra_ldflags=['-ldl'],with_cuda=True)
 import sgl_kernel.flash_ops
 bridge.fa_bind(sgl_kernel.flash_ops.__file__)
 red=load(_BUNDLE['reduce'],'void row_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',['row_forward'])
 sms=torch.cuda.get_device_properties('cuda').multi_processor_count;memo={}
 def arms(data):
  c=data['config'];B,L,W=c['batch'],c['q_len'],c['window']
  if L!=8 or c['num_q_heads']!=16:return ['A','A2','S']
  if (B,W)==(1,0):return ['A','A2','S']+[f'M{s}' for s in (32,64,128)]+[f'R{s}' for s in (32,64,128)]+['R64w4','D64']
  if W and B in (1,16):
   ss=(1,2,4,8,16) if B==1 else (1,2,4,8)
   return ['A','A2','S']+[f'M{s}' for s in ss]+[f'R{s}' for s in ss]+['R4w4','D4']
  if not W and B in (16,32):return ['A','A2','S','R','Rw4','G4','F4','G8','F8']
  return ['A','A2','S']
 def setup(data):
  key=data['qkv'].data_ptr()
  if key not in memo:
   B=data['config']['batch'];memo[key]=torch.empty((max(sms,8*B)+B)*128*68+256,device='cuda')
  return memo[key]
 def pair_of(data):
  c=data['config'];return data.get('_partial_pair',(data['qkv'].reshape(1,c['batch']*c['q_len'],-1),1))
 def vq(data,mod,out,part,cs=None):
  c=data['config'];B,L=c['batch'],c['q_len'];P,S=pair_of(data);split=S if P.dtype==torch.float32 else 0;raw=P if split else P[0]
  stage=4 if arch=='90a' else 2;grid=sms;tiles=8
  if cs is not None:
   cs=cs if arch=='90a' or getattr(mod,'_p5_cluster_control',False) else 1;grid=cs*B;tiles=cs
  args=[raw.data_ptr()]+[data[k].data_ptr() for k in ('cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
  args +=[0,part.data_ptr(),out.data_ptr(),B,L,16,2,raw.stride(-2),data['page_table'].shape[1],c['page_size'],0,split,tiles,grid,stage,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,P.stride(0) if split else 0,1,torch.cuda.current_stream().cuda_stream]
  mod.vq8_forward(*args)
 def row(fa,data,out,oa=0,lse=0,gate=0,prefix=0,part=0,ns=0,warps=1):
  c=data['config'];P,S=pair_of(data);split=S if P.dtype==torch.float32 else 0;raw=P if split else P[0]
  red.row_forward(part,prefix,oa,lse,raw.data_ptr(),gate,data['v_scale'].data_ptr(),out.data_ptr(),c['batch'],c['q_len'],16,2,ns,raw.stride(-2),split,P.stride(0) if split else 0,int(fa),warps,torch.cuda.current_stream().cuda_stream)
 def call(name,data):
  c=data['config'];B,L=c['batch'],c['q_len'];W=c['window'];d=dict(data,config=dict(c,tokens=B*L));P,S=pair_of(data)
  if name=='S' and '_partial_pair' not in data:return alt.kernel_fn(data)
  if name in ('A','A2','S'):
   owner=alt if name=='S' else base
   if '_partial_pair' in data:
    key=str(P.device),c['rope_theta']
    if key not in owner._ROOF_STATE:owner.kernel_fn(data)
    st=owner._ROOF_STATE[key];setting=owner.BEST7_SETTINGS[(B,L,W)];glue=setting[0][0] if isinstance(setting[0],tuple) else setting[0];return owner._best7_attention((P,S),d,st,dict(glue=glue,attention=setting[1])).view(B,L,-1)
   return owner.kernel_fn(data)
  out=torch.empty((B,L,1024),device='cuda',dtype=torch.bfloat16)
  if L==8 and ((W and B in (1,16)) or (not W and B==1)):
   a=name.split('w')[0];ns=int(a[1:]);warps=4 if 'w4' in name else 1
   query,gate=base.qk_prep((P,S),d,dict(inv_freq=base._inv_freq(c['rope_theta'],'cuda')),q_fp8=True)
   if name.startswith('M'):return sched['_p5b_sched_fa'](query,gate,d,ns,None).view(B,L,-1)
   effective_window=bool(W and W<data['page_table'].shape[1]*c['page_size'])
   meta=sched['_P5B_FA_META'][B,L,effective_window,ns];faout=torch.empty((B*L,16,64),device='cuda',dtype=torch.bfloat16);lse=torch.empty((16,B*L),device='cuda');oa=torch.empty((ns,16,B*L,64),device='cuda');la=torch.empty((ns,16,B*L),device='cuda')
   if arch=='90a':
    bridge.fa_main(query,data['k_cache'],data['v_cache'],faout,lse,oa,la,data['cache_seqlens'],data['cu_seqlens_q'],data['page_table'],meta,data['k_scale'],data['v_scale'],L,W,ns,c['sm_scale']*c['qk_scale_factor'],int(name.startswith('D'))+(2 if c['kv_scale']==1 else 0),torch.cuda.current_stream().cuda_stream)
   else:
    bridge.fa_probe(query,data['k_cache'],data['v_cache'],faout,lse,oa,la,data['cache_seqlens'],data['cu_seqlens_q'],data['page_table'],meta,data['k_scale'],data['v_scale'],L,W,ns,c['sm_scale']*c['qk_scale_factor'])
    globals()['CHAIN_LOCAL'].fa_main(base,query,d,ns,oa,la)
    if name.startswith('D') or ns==1:
     # Local bridge glue needs an ordinary normalized attention output. The
     # exact own reducer first merges stand-in partials; undo gate/norm is not
     # needed because D is only a native launcher control on Hopper.
     if name.startswith('D'):row(True,d,out,oa.data_ptr(),la.data_ptr(),gate.data_ptr(),meta[-1:].data_ptr(),ns=ns,warps=1);return out
     # One-split FA3 main returns BF16 before the fused norm/gate.
     faout.copy_(oa[0].permute(1,0,2).to(torch.bfloat16))
   if name.startswith('D'):
    base._launch(base._norm_gate_kernel,(B*L,),faout,gate,out,base.RMS_EPS,HQ=16,D=64,num_warps=4);return out
   row(True,d,out,(faout if ns==1 else oa).data_ptr(),la.data_ptr(),gate.data_ptr(),meta[-1:].data_ptr(),ns=ns,warps=warps);return out
  part=setup(data)
  if name in ('R','Rw4'):
   vq(data,vmods['N'],out,part);prefix=part[(sms+B)*128*68:]
   row(False,d,out,part=part.data_ptr(),prefix=prefix.data_ptr(),warps=4 if name=='Rw4' else 1);return out
  cs=int(name[1:]);vq(data,vmods[name[0]],out,part,cs);return out
 def graph(fn):
  s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(s):
   for _ in range(2):out=fn()
  torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(g):out=fn()
  assert guard.graph_kernels(g.raw_cuda_graph())>0,'empty capture'
  return g,out
 e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 def replay(g):
  lab_cold_l2();e0.record();g.replay();e1.record();e1.synchronize();return e0.elapsed_time(e1)*1000
 x=torch.zeros(1,device='cuda');empty,_=graph(lambda:x.add_(1));fixed=statistics.median(replay(empty) for _ in range(16))
 ref_orig=ck['ref_kernel'];cache={}
 def ref_cached(data,query_dtype=None,new_kv=None):
  key=(query_dtype,None if new_kv is None else tuple(t.view(torch.uint8).cpu().numpy().tobytes() for t in new_kv))
  if key not in cache:cache[key]=ref_orig(data,query_dtype,new_kv)
  return cache[key]
 ck['ref_kernel']=ref_cached
 def check(data,y,label,group):
  ok,msg=ck['check_implementation'](data,ck['collect'](data,y));counts[group]+=1
  if not ok:fail.append(label+':'+msg)
 def discard():memo.clear();cache.clear();gc.collect();torch.cuda.empty_cache()
 official=[dict(batch=1,ctx=10,layer='sliding'),dict(batch=3,ctx=2047,layer='sliding'),dict(batch=5,ctx=5000,layer='sliding',q_len=8,spread='mixed',page=1),dict(batch=4,ctx=3000,layer='full'),dict(batch=8,ctx=8192,heads='dp',layer='full',spread='mixed'),dict(batch=2,ctx=100000,layer='full',q_len=8,page=1),dict(batch=4,ctx=6000,layer='sliding',dist='edges',page=1),dict(batch=4,ctx=6000,layer='full',q_len=8,dist='edges'),dict(batch=1,ctx=1048576,layer='full',q_len=8),dict(batch=3,ctx=4000,layer='sliding',q_len=8,page=1,kv_scale=.015625),dict(batch=2,ctx=9000,layer='full',kv_scale=.015625),dict(batch=4,ctx=80000,layer='full',spread='mixed',dist='soft'),dict(batch=4,ctx=6000,layer='sliding',q_len=8,dist='soft',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='soft'),dict(batch=4,ctx=80000,heads='tp8',layer='full',spread='mixed',dist='coherent',page=1),dict(batch=2,ctx=20000,heads='dp',layer='full',q_len=8,dist='coherent')]
 stress=[dict(batch=b,q_len=8,ctx=129,layer=layer,page=1,dist='edges',kv_scale=k) for b,layer in ((1,'sliding'),(16,'sliding'),(1,'full'),(16,'full'),(32,'full')) for k in (1.,.015625)]
 for group,specs in [('official',official),('stress',stress)]:
  for i,spec in enumerate(specs):
   print('PHASES_GATE',group,i,flush=True)
   data=ck['generate_input'](seed=2930123+i,**dict({'heads':'tp8'},**spec));setup(data)
   for name in arms(data):
    fn=lambda:call(name,data);y=fn();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:eager',group)
    g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:graph',group);del g,y
   data=None;discard()
 # Whole-layer QKV comes from strided FP32 split-K partials, not only BF16.
 for pi,(B,L,W) in enumerate(CASES+((16,8,0),)):
  splits=4 if pi==5 else 3
  data=ck['generate_input'](batch=B,q_len=L,ctx=257,layer='sliding' if W else 'full',page=1,heads='tp8',dist='coherent',seed=6917723+pi,kv_scale=.015625)
  t=B*L;width=data['qkv'].shape[-1]
  storage=torch.randn((splits,t*width+64),device='cuda',dtype=torch.float32)*.03125
  partial=storage[:,:t*width].view(splits,t,width);partial[-1].copy_(data['qkv'].reshape(t,width).float()-partial[:-1].sum(0))
  data['qkv']=partial.sum(0).to(torch.bfloat16).reshape_as(data['qkv']);data['_partial_pair']=(partial,splits)
  setup(data)
  for name in arms(data):
   fn=lambda:call(name,data);y=fn();torch.cuda.synchronize();check(data,y,f'partial{pi}{name}:eager','partial')
   g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(data,y,f'partial{pi}{name}:graph','partial');del g,y
  data=partial=storage=None;discard()
 assert not fail,('TASK',counts,fail)
 summary={};times={};graph_nodes={};profiles={};names=[]
 activity=None
 if arch=='90a':
  activity=load_inline(name='p5_phases_activity_'+hashlib.sha256(_BUNDLE['activity'].encode()).hexdigest()[:12],cpp_sources=_BUNDLE['activity'],functions=['profile_begin','profile_end'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'])
 for B,L,W in CASES:
  label=f'{B}x{L}'+('s' if W else 'f');samples=[]
  for si,seed in enumerate(SEEDS):
   if time.monotonic()>deadline-8:raise TimeoutError((label,si,counts))
   data=ck['generate_input'](batch=B,q_len=L,ctx=80000,layer='sliding' if W else 'full',page=1,heads='tp8',dist='soft' if W else 'coherent',spread='agentic' if B>1 else 'fixed',seed=seed);setup(data)
   order=arms(data);order=order[si%len(order):]+order[:si%len(order)];graphs={}
   for name in order:
    g,y=graph(lambda n=name:call(n,data));y.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(data,y,label+name,'scored');graphs[name]=(g,y)
    nodes=guard.graph_kernels(g.raw_cuda_graph());assert nodes>0,(name,nodes);graph_nodes[label+':'+name]=nodes
   if si==0 and activity:
    for n,(g,_) in graphs.items():
     if n not in ('A','A2','S','M4','M64','D4','D64','R1','R4','R64','R64w4','R4w4','R','Rw4','G4','F4','G8','F8'):continue
     lab_cold_l2();torch.cuda.synchronize();activity.profile_begin();g.replay();torch.cuda.synchronize();rows=activity.profile_end();profile=[]
     origin=min(v[0] for _,v in rows)
     for kn,v in rows:
      if kn not in names:names.append(kn)
      profile.append([names.index(kn),v[0]-origin,v[1]-origin,*v[7:13]])
     profiles[label+':'+n]=profile
   per={n:[] for n in order}
   for ri in range(8):
    rotated=order[ri%len(order):]+order[:ri%len(order)]
    if ri&1:rotated.reverse()
    for n in rotated:per[n].append(replay(graphs[n][0])-fixed)
   assert all(min(v)>0 for v in per.values()),(label,per)
   samples.append({n:round(statistics.mean(v),3) for n,v in per.items()})
   graphs=data=None;discard()
  print('PHASES_TIMED',label,counts,flush=True)
  times[label]=samples;ratios={}
  for n in samples[0]:
   if n=='A':continue
   rr=[s[n]/s['A'] for s in samples];r=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5;ratios[n]=dict(r=r,se=se,lo=r-2.364624251*se,hi=r+2.364624251*se)
  summary[label]=dict(control_us=statistics.mean(s['A'] for s in samples),ratios=ratios)
 assert not fail,('TASK',counts,fail)
 result=dict(profiles=profiles,profile_names=names,complete=True,graph_nodes=graph_nodes,counts=counts,failures=fail,seed_us=times,seeds=SEEDS,

  vq_resources={n:m.vq8_resources(4 if arch=='90a' else 2) for n,m in vmods.items()},elapsed=time.monotonic()-start,
  base_sha256=hashlib.sha256(_BUNDLE['source'].encode()).hexdigest(),schedule='Shape-only B/L/window/explicit split metadata, no length values; direct exported FA3 main binding; distributed row/DSM merge')
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8400,len(blob)
 return ['META S19 FA3 schedule+exported-main parallel norm/merge; native full DSM cluster merge vs fixed-geometry control;8freshseeds8cold,TASK,A:A,nodecounts','RESULT_XZ85 '+blob]
