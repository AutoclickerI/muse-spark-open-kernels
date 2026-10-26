SEEDS=tuple(3671223+7919*i for i in range(8))
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
 base=module('S18',_BUNDLE['source']+_BUNDLE['adapter']);base.ATTN_PORT_CUDA_SRC=_BUNDLE['q1'];base.VQ8_CUDA=_BUNDLE['A_vq']
 if install:=globals().get('ROOF_LOCAL_INSTALL'):install(base)
 qmods={'A':load(_BUNDLE['q1'],_BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])}
 for n in ('C3','C4'):qmods[n]=load(_BUNDLE[n],_BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
 base._ATTN_PORT_EXT=qmods['A']
 base._PP_MOD={False:load(_BUNDLE['PARTIAL_CUDA'],_BUNDLE['PARTIAL_CPP'],['attn_forward','occupancy','max_cluster'])}
 vmods={}
 if arch=='90a':
  for n in ('A_vq','F_vq','B_vq'):vmods[n]=load(_BUNDLE[n],_BUNDLE.get(n+'_CPP',_BUNDLE['VQ8_CPP']),['vq8_forward','vq8_resources'])
 else:
  # Every edited softmax and launch runs locally; only Hopper instructions and
  # oversized V rings use the established exact arithmetic stand-in.
  local_n128=globals()['FIXED_LOCAL_N128']
  for n in ('A_vq','F_vq'):
   m=module(n,_BUNDLE['source']);m.VQ8_CUDA=_BUNDLE[n];m.VQ8_CPP=_BUNDLE.get(n+'_CPP',_BUNDLE['VQ8_CPP']);local_n128.install(m);vmods[n]=m._vq8_module()
  c=_BUNDLE['B_vq'].replace('w=threadIdx.x>>5;','w=(threadIdx.x>>5)&3;').replace('return CLKPAD + 25600 + ST * 49152;','return CLKPAD + 25600 + ST * 16384;')
  vmods['B_vq']=load(c,_BUNDLE['VQ8_CPP'],['vq8_forward','vq8_resources'])
 base._VQ8_MOD=vmods['A_vq'];base._vq8_module=lambda:vmods['A_vq']
 last=load(_BUNDLE['last'],_BUNDLE['last_CPP'],['sliding_forward','sliding_resources'])
 sms=torch.cuda.get_device_properties('cuda').multi_processor_count
 counters={};memo={}
 def arms(data):
  c=data['config'];B,L=c['batch'],c['q_len'];full=not c['window'] and c['num_q_heads']==16
  if full and B==1:return ['A','A2']+[f'L{nk}s{ns}' for nk in (64,128) for ns in (16,32,64,128,256)]
  if full and L==8 and B in (16,32):return ['A','A2','Ffree','Fpaid']
  if full and L==1 and B in (16,32):return ['A','A2','C3','C4']
  return ['A','A2']
 def vq(data,mod,out,part,fixed=None):
  stream=torch.cuda.current_stream(data['qkv'].device).cuda_stream
  c=data['config'];B,L=c['batch'],c['q_len'];stage=4 if arch=='90a' else 2
  args=[data[k].data_ptr() for k in ('qkv','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
  args +=[0,part.data_ptr(),out.data_ptr(),B,L,c['num_q_heads'],c['num_kv_heads'],data['qkv'].stride(-2),data['page_table'].shape[1],c['page_size'],c['window'],0,8,sms,stage,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,0,1,stream]
  if fixed is not None:args.append(fixed.data_ptr())
  mod.vq8_forward(*args)
 def setup(data):
  c=data['config'];B,L=c['batch'],c['q_len'];key=data['qkv'].data_ptr()
  if key not in memo:
   part=torch.empty((sms+B)*128*68+256,device='cuda');ref=torch.empty((B,L,16),device='cuda')
   memo[key]=(part,ref)
   if 'Ffree' in arms(data):vq(data,vmods['B_vq'],ref,part)
  return memo[key]
 def call(name,data):
  stream=torch.cuda.current_stream(data['qkv'].device).cuda_stream
  c=data['config'];B,L=c['batch'],c['q_len'];HQ,HK=c['num_q_heads'],c['num_kv_heads'];q=data['qkv'];part,ref=setup(data)
  if name=='A2':name='A'
  if name=='A':return base.kernel_fn(data)
  out=torch.empty((B,L,HQ*64),device=q.device,dtype=torch.bfloat16)
  if name.startswith('F'):
   if name=='Fpaid':vq(data,vmods['B_vq'],ref,part)
   vq(data,vmods['F_vq'],out,part,ref);return out
  if name.startswith('C'):
   mod=qmods[name];KVG=HK//2;occ=(max(1,mod.occupancy(1)),{cs:mod.max_cluster(1,cs) for cs in range(1,17)})
   ns,cs=base._attn_main_plan(B,KVG,data['page_table'].shape[1]*c['page_size'],1,0,sms,occ);rows=B*KVG*(ns//cs)*16
   po=torch.empty(rows*64,device='cuda');ml=torch.empty(rows*2,device='cuda');p1=po
   ckey=(name,B,KVG,ns,cs)
   if ckey not in counters:counters[ckey]=torch.zeros(B*KVG*cs,device='cuda',dtype=torch.int32)
   args=[data[k].data_ptr() for k in ('qkv','positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')]
   args +=[0,po.data_ptr(),ml.data_ptr(),counters[ckey].data_ptr(),p1.data_ptr(),1,out.data_ptr(),B,L,q.stride(-2),HQ,HK,data['page_table'].shape[1],c['page_size'],0,ns,cs,c['sm_scale']*c['qk_scale_factor']*1.4426950408889634]
   mod.attn_forward(*args,stream);return out
  nk,ns=map(int,name[1:].split('s'));groups=B*HK
  ckey=('L',B,L,ns,nk)
  if ckey not in counters:counters[ckey]=torch.zeros(groups*32,device='cuda',dtype=torch.int32)
  p=torch.empty(groups*ns*(L*8)*68,device='cuda')
  last.sliding_forward(*[data[k].data_ptr() for k in ('qkv','positions','cache_loc','page_table','seq_lens','k_cache','v_cache','k_scale','v_scale')],0,out.data_ptr(),B,L,HQ,HK,q.stride(-2),data['page_table'].shape[1],c['page_size'],0,0,0,1,nk,1,q.shape[-1],c['sm_scale']*c['qk_scale_factor']*1.4426950408889634,stream,p.data_ptr(),counters[ckey].data_ptr(),ns)
  return out
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
   data=ck['generate_input'](seed=1930117+i,**dict({'heads':'tp8'},**spec));setup(data)
   for name in arms(data):
    fn=lambda:call(name,data);y=fn();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:eager',group)
    g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(data,y,f'{group}{i}{name}:graph',group);del g,y
   data=None;discard()
 assert not fail,('TASK',counts,fail)
 summary={};times={};prepass={};graph_nodes={}
 for B,L in CASES:
  label=f'{B}x{L}f';samples=[];bound_us=[]
  for si,seed in enumerate(SEEDS):
   if time.monotonic()>deadline-8:raise TimeoutError((label,si,counts))
   data=ck['generate_input'](batch=B,q_len=L,ctx=80000,layer='full',page=1,heads='tp8',dist='coherent',spread='agentic' if B>1 else 'fixed',seed=seed);setup(data)
   order=arms(data);order=order[si%len(order):]+order[:si%len(order)];graphs={}
   for name in order:
    g,y=graph(lambda n=name:call(n,data));y.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(data,y,label+name,'scored');graphs[name]=(g,y)
    nodes=guard.graph_kernels(g.raw_cuda_graph());assert nodes>=({'Ffree':2,'Fpaid':4}.get(name,1)),(name,nodes);graph_nodes[label+':'+name]=nodes
   pp=None
   if 'Fpaid' in order:
    part,ref=setup(data);pp,_=graph(lambda:vq(data,vmods['B_vq'],ref,part));bound_us.append(round(statistics.mean(replay(pp)-fixed for _ in range(8)),3))
   per={n:[] for n in order}
   for ri in range(8):
    rotated=order[ri%len(order):]+order[:ri%len(order)]
    if ri&1:rotated.reverse()
    for n in rotated:per[n].append(replay(graphs[n][0])-fixed)
   assert all(min(v)>0 for v in per.values()),(label,per)
   samples.append({n:round(statistics.mean(v),3) for n,v in per.items()})
   graphs=pp=data=None;discard()
  times[label]=samples;prepass[label]=bound_us;ratios={}
  for n in samples[0]:
   if n=='A':continue
   rr=[s[n]/s['A'] for s in samples];r=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5;ratios[n]=dict(r=r,se=se,lo=r-2.364624251*se,hi=r+2.364624251*se)
  summary[label]=dict(control_us=statistics.mean(s['A'] for s in samples),ratios=ratios)
 assert not fail,('TASK',counts,fail)
 result=dict(complete=True,graph_nodes=graph_nodes,counts=counts,failures=fail,seed_us=times,summary=summary,prepass_us=prepass,seeds=SEEDS,
  q1_resources={n:dict(ctas=m.occupancy(1),clusters2=m.max_cluster(1,2)) for n,m in qmods.items()},
  vq_resources={n:m.vq8_resources(4 if arch=='90a' else 2) for n,m in vmods.items()},elapsed=time.monotonic()-start,
  oracle='Ffree bound pass outside measured call: ceiling only, not nominatable. Fpaid includes it.')
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<8400,len(blob)
 return ['META S18+R8+D; exact all-key bound oracle and PAID prepass; lastCTA full L1/L8; compactST3/4;allTASK,A:A8seeds8cold','RESULT_XZ85 '+blob]
