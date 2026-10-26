LAB_RESOURCES={'compile_s':100,'run_s':400,'output_bytes':7800,'context_poison':'none'}
SEEDS=tuple(4671259+7919*i for i in range(8))
CASES=tuple((b,l) for l in (1,8) for b in (1,16,32))
def module(name,source):
 f='<p5-fixed-'+name+'>';linecache.cache[f]=(len(source),None,source.splitlines(True),f)
 m=types.ModuleType('p5_fixed_'+name);m.__file__=f;exec(compile(source,f,'exec'),m.__dict__);return m

def lab(deadline,weights):
 start=time.monotonic();fail=[];counts=dict(official=0,scored=0,stress=0)
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
 base=module('S19',_BUNDLE['source']+_BUNDLE['adapter']);base.ATTN_PORT_CUDA_SRC=_BUNDLE['Aq'];base.VQ8_CUDA=_BUNDLE['Av']
 if install:=globals().get('ROOF_LOCAL_INSTALL'):install(base)
 qmods={'A':load(_BUNDLE['Aq'],_BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])}
 for n in ('C3','T8','T16'):qmods[n]=load(_BUNDLE[n],_BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
 base._ATTN_PORT_EXT=qmods['A']
 pm=load(_BUNDLE['P'],_BUNDLE['PARTIAL_CPP'],['attn_forward','occupancy','max_cluster']);base._PP_MOD={False:pm}
 vmods={}
 for n in ('Av','E25','E50'):
  if arch=='90a':vmods[n]=load(_BUNDLE[n],_BUNDLE['VQ8_CPP'],['vq8_forward','vq8_resources'])
  else:
   m=module(n,_BUNDLE['source']);m.VQ8_CUDA=_BUNDLE[n];globals()['FIXED_LOCAL_N128'].install(m);vmods[n]=m._vq8_module()
 base._VQ8_MOD=vmods['Av'];base._vq8_module=lambda:vmods['Av']
 sms=torch.cuda.get_device_properties('cuda').multi_processor_count
 counters={};memo={}
 # Shape-only legal scheduling constants, not measured sequence statistics.
 schedules={v:torch.tensor([v,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],device='cuda',dtype=torch.int32) for v in (1,8,16,32,64,128)}
 def arms(data):
  c=data['config'];B,L=c['batch'],c['q_len'];full=not c['window'] and c['num_q_heads']==16
  if full and (B,L)==(1,1):return ['A','A2']+[f'P{ns}c{cs}' for ns,cs in ((64,8),(128,8),(128,16),(256,8),(256,16),(384,16))]
  if full and (B,L)==(1,8):return ['A','A2']+['M'+str(v) for v in (1,8,16,32,64,128)]+['U64']
  if full and L==8 and B in (16,32):return ['A','A2','E25','E50']
  if full and L==1 and B in (16,32):return ['A','A2','C3','T8','T16']
  return ['A','A2']
 def vq(data,mod,out,part):
  stream=torch.cuda.current_stream(data['qkv'].device).cuda_stream
  c=data['config'];B,L=c['batch'],c['q_len'];stage=4 if arch=='90a' else 2
  args=[data[k].data_ptr() for k in ('qkv','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
  args +=[0,part.data_ptr(),out.data_ptr(),B,L,c['num_q_heads'],c['num_kv_heads'],data['qkv'].stride(-2),data['page_table'].shape[1],c['page_size'],c['window'],0,8,sms,stage,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,0,1,stream]
  mod.vq8_forward(*args)
 def setup(data):
  c=data['config'];B,L=c['batch'],c['q_len'];key=data['qkv'].data_ptr()
  if key not in memo:
   part=torch.empty((sms+B)*128*68+256,device='cuda');ref=torch.empty((B,L,16),device='cuda')
   memo[key]=(part,ref)

  return memo[key]
 def call(name,data):
  stream=torch.cuda.current_stream(data['qkv'].device).cuda_stream
  c=data['config'];B,L=c['batch'],c['q_len'];HQ,HK=c['num_q_heads'],c['num_kv_heads'];q=data['qkv'];part,ref=setup(data)
  if name in ('A','A2'):return base.kernel_fn(data)
  out=torch.empty((B,L,HQ*64),device=q.device,dtype=torch.bfloat16)
  if name.startswith('E'):vq(data,vmods[name],out,part);return out
  if name.startswith(('M','U')):
   from sgl_kernel.flash_attn import flash_attn_with_kvcache
   ns=int(name[1:]);d=dict(data,config=dict(c,tokens=B*L));pair=(q.reshape(1,B*L,-1),1)
   st=dict(inv_freq=base._inv_freq(c['rope_theta'],'cuda'))
   query,gate=base.qk_prep(pair,d,st,q_fp8=True)
   descale={} if c['kv_scale']==1 else dict(k_descale=data['k_scale'].reshape(1,1).expand(B,HK),v_descale=data['v_scale'].reshape(1,1).expand(B,HK))
   a=flash_attn_with_kvcache(q=query,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=L,softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(-1,-1),scheduler_metadata=schedules[ns] if name.startswith('M') else None,num_splits=ns,**descale)
   if isinstance(a,tuple):a=a[0]
   base._launch(base._norm_gate_kernel,(B*L,),a,gate,out,base.RMS_EPS,HQ=HQ,D=64,num_warps=4)
   return out
  partial=name.startswith('P');mod=pm if partial else qmods[name]
  occ=(max(1,mod.occupancy(1)),{cs:mod.max_cluster(1,cs) for cs in range(1,17)})
  if partial:
   ns,cs=map(int,name[1:].split('c'))
   if arch!='90a' and cs>8:
    cs=8;ns=min(ns,32*cs) # maintain native <=32clusters/request invariant in local stand-in
  else:ns,cs=base._attn_main_plan(B,HK//2,data['page_table'].shape[1]*c['page_size'],1,0,sms,occ)
  rows=B*(HK//2)*(ns//cs)*16;po=torch.empty(rows*64,device='cuda');ml=torch.empty(rows*2,device='cuda');ckey=(name,B,ns,cs)
  if ckey not in counters:counters[ckey]=torch.zeros(B*(HK//2)*cs,device='cuda',dtype=torch.int32)
  args=[data[k].data_ptr() for k in ('qkv','positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
  args +=[0,po.data_ptr(),ml.data_ptr(),counters[ckey].data_ptr(),po.data_ptr(),1,out.data_ptr(),B,L,q.stride(-2),HQ,HK,data['page_table'].shape[1],c['page_size'],0,ns,cs,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634]
  if partial:args +=[0,0,q.shape[-1]]
  mod.attn_forward(*args,stream);return out
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
 stress=[dict(batch=b,q_len=l,ctx=129,layer='full',page=1,dist='edges',kv_scale=k) for b in (1,16,32) for l in (1,8) for k in (1.,.015625)]
 for group,specs in [('official',official),('stress',stress)]:
  for i,spec in enumerate(specs):
   print('PHASES_GATE',group,i,flush=True)
   data=ck['generate_input'](seed=1930117+i,**dict({'heads':'tp8'},**spec));setup(data)
   for name in arms(data):
    fn=lambda:call(name,data);y=fn();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:eager',group)
    g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:graph',group);del g,y
   data=None;discard()
 assert not fail,('TASK',counts,fail)
 summary={};times={};graph_nodes={};profiles={};names=[]
 activity=None
 if arch=='90a':
  activity=load_inline(name='p5_phases_activity_'+hashlib.sha256(_BUNDLE['activity'].encode()).hexdigest()[:12],cpp_sources=_BUNDLE['activity'],functions=['profile_begin','profile_end'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_include_paths=['/usr/local/cuda/include'],extra_ldflags=['-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcupti','-lcudart'])
 for B,L in CASES:
  label=f'{B}x{L}f';samples=[]
  for si,seed in enumerate(SEEDS):
   if time.monotonic()>deadline-8:raise TimeoutError((label,si,counts))
   data=ck['generate_input'](batch=B,q_len=L,ctx=80000,layer='full',page=1,heads='tp8',dist='coherent',spread='agentic' if B>1 else 'fixed',seed=seed);setup(data)
   order=arms(data);order=order[si%len(order):]+order[:si%len(order)];graphs={}
   for name in order:
    g,y=graph(lambda n=name:call(n,data));y.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(data,y,label+name,'scored');graphs[name]=(g,y)
    nodes=guard.graph_kernels(g.raw_cuda_graph());assert nodes>0,(name,nodes);graph_nodes[label+':'+name]=nodes
   if si==0 and activity:
    for n,(g,_) in graphs.items():
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
 result=dict(profiles=profiles,profile_names=names,complete=True,graph_nodes=graph_nodes,counts=counts,failures=fail,seed_us=times,summary=summary,seeds=SEEDS,
  q1_resources={n:dict(ctas=m.occupancy(1),clusters2=m.max_cluster(1,2)) for n,m in qmods.items()},
  vq_resources={n:m.vq8_resources(4 if arch=='90a' else 2) for n,m in vmods.items()},elapsed=time.monotonic()-start,
  base_sha256=hashlib.sha256(_BUNDLE['source'].encode()).hexdigest(),schedule='explicit split count, M1, batch0, heads_l2=1; fresh lengths in main; combine resets semaphore')
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8400,len(blob)
 return ['META S19; static legal FA3 schedule; D pure-register poly; C3 vector page IDs; d1 split geometry;TASK/poison/capture/A:A8seeds8cold','RESULT_XZ85 '+blob]
