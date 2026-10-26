LAB_RESOURCES={'compile_s':350,'run_s':460,'output_bytes':11200,'context_poison':'none'}
SEEDS=tuple(8671259+7919*i for i in range(8))
CASES=tuple((b,l,w) for l in (1,8) for b in (1,16,32) for w in (2048,0))
ABI={(1,1,2048):('f',1,1280),(1,1,0):('f',1,1280),(16,1,2048):('f',10,1280),(16,1,0):('f',7,2304),(32,1,2048):('f',1,2304),(32,1,0):('f',1,1280),(1,8,2048):('f',1,2304),(1,8,0):('f',1,2304),(16,8,2048):('f',3,2304),(16,8,0):('f',3,2304),(32,8,2048):('b',1,2304),(32,8,0):('b',1,2304)}
def module(name,src):
 f='<p5b-public-'+name+'>';linecache.cache[f]=(len(src),None,src.splitlines(True),f);m=types.ModuleType('p5b_public_'+name);m.__file__=f;exec(compile(src,f,'exec'),m.__dict__);return m

def lab(deadline,weights):
 global base,op,lee,pack
 start=time.monotonic();checks={};fail=[]
 ck=module('TASK',BUNDLE['checker']).__dict__;arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
 if arch!='90a':ck['_fa3_scheduler_metadata']=lambda *a,**kw:None
 base=module('S19',BUNDLE['source']);op=module('operator',BUNDLE['operator']);lee=module('lee',BUNDLE['lee']);pack=module('input',BUNDLE['input'])
 if hook:=globals().get('PUBLIC_LOCAL_INSTALL'):hook(base,op,lee)
 from torch.utils.cpp_extension import load_inline
 def load(cu,cpp,functions):
  tag=hashlib.sha256((cu+cpp).encode()).hexdigest()[:14]
  return load_inline(name='p5_public_'+tag+'_'+arch,cpp_sources='#include <pybind11/stl.h>\n'+cpp,cuda_sources=cu,functions=functions,no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
 guard=load(BUNDLE['guard_cu'],'#include <cstdint>\nint64_t graph_kernels(int64_t);',['graph_kernels'])
 base._ATTN_PORT_EXT=load(BUNDLE['ATTN_PORT_CUDA_SRC'],BUNDLE['ATTN_PORT_CPP_SRC'],['attn_forward','occupancy','max_cluster'])
 base._PP_MOD={False:load(BUNDLE['PARTIAL_CUDA'],BUNDLE['PARTIAL_CPP'],['attn_forward','occupancy','max_cluster'])}
 if arch=='90a':
  vm=load(BUNDLE['VQ8_CUDA'],BUNDLE['VQ8_CPP'],['vq8_forward','vq8_resources']);base._VQ8_MOD=vm;base._vq8_module=lambda:vm
 else:vm=base._vq8_module()
 op.setup('cuda');lee.setup('cuda')
 def graph(fn):
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   for _ in range(2):out=fn()
  torch.cuda.current_stream().wait_stream(stream);g=torch.cuda.CUDAGraph(keep_graph=True)
  with torch.cuda.graph(g):out=fn()
  assert guard.graph_kernels(g.raw_cuda_graph())>0,'empty capture'
  return g,out
 e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 def replay(g):
  lab_cold_l2();e0.record();g.replay();e1.record();e1.synchronize();return round(e0.elapsed_time(e1)*1000000)
 x=torch.zeros(1,device='cuda');empty,_=graph(lambda:x.add_(1));fixed=statistics.median(replay(empty) for _ in range(24))
 ref=ck['ref_kernel'];refs={}
 def reference(d,query_dtype=None,new_kv=None):
  key=(query_dtype,None if new_kv is None else tuple(t.view(torch.uint8).cpu().numpy().tobytes() for t in new_kv))
  if key not in refs:refs[key]=ref(d,query_dtype,new_kv)
  return refs[key]
 ck['ref_kernel']=reference
 def check(d,y,label):
  ok,msg=ck['check_implementation'](d,ck['collect'](d,y));checks[label.split(':')[0]]=checks.get(label.split(':')[0],0)+1
  if not ok:fail.append(label+':'+msg)
 def pair(d,abi=None):
  c=d['config'];b,l=c['batch'],c['q_len'];q=d['qkv'];w=q.shape[-1]
  if abi is None:return q.reshape(1,b*l,w),1
  typ,s,w=abi
  if typ=='b':return q.reshape(1,b*l,w),1
  p=torch.randn((s,b*l,w),device='cuda')*.03125
  v=q[...,:w].reshape(b*l,w).float();p[-1].copy_(v-p[:-1].sum(0))
  v=p[0].clone()
  for i in range(1,s):v+=p[i]
  d['qkv']=q.clone();d['qkv'][...,:w]=v.to(torch.bfloat16).reshape(b,l,w)
  if w==1280:d['qkv'][...,1280:]=float('inf')
  return p,s
 def cleanup():
  refs.clear();gc.collect();torch.cuda.empty_cache()
 official=BUNDLE['official']
 for i,spec in enumerate(official):
  print('PUBLIC_GATE',i,flush=True);d=ck['generate_input'](seed=8145723+i,**spec);p=pair(d)
  for name in ('A','A2','O','L'):
   fn=lambda:call(name,p,d);y=fn();torch.cuda.synchronize();check(d,y,'official:'+str(i)+name+':eager')
   g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(d,y,'official:'+str(i)+name+':graph');del g,y
  d=p=None;cleanup()
 for i,(b,l,w) in enumerate(CASES):
  for scale in (1.,.015625):
   d=ck['generate_input'](batch=b,q_len=l,ctx=257,page=1,heads='tp8',layer='sliding' if w else 'full',dist='edges',kv_scale=scale,seed=9148123+i);p=pair(d,ABI[b,l,w])
   for name in ('A','A2','O','L'):
    fn=lambda:call(name,p,d);y=fn();torch.cuda.synchronize();check(d,y,'ABI:'+str(i)+name+':eager')
    g,y=graph(fn);y.fill_(float('nan'));g.replay();g.replay();torch.cuda.synchronize();check(d,y,'ABI:'+str(i)+name+':graph');del g,y
   d=p=None;cleanup()
 assert not fail,('TASK',checks,fail)
 rounds={};nodes={}
 for b,l,w in CASES:
  label=f'{b}x{l}'+('s' if w else 'f');samples=[]
  for si,seed in enumerate(SEEDS):
   assert time.monotonic()<deadline-8,(label,si,'deadline')
   d=ck['generate_input'](batch=b,q_len=l,ctx=80000,page=1,heads='tp8',layer='sliding' if w else 'full',dist='soft' if w else 'coherent',spread='agentic' if b>1 else 'fixed',seed=seed);p=pair(d,ABI[b,l,w]);graphs={}
   names=('A','A2','O','L')
   for name in names:
    g,y=graph(lambda n=name:call(n,p,d));y.fill_(float('nan'));g.replay();torch.cuda.synchronize();check(d,y,'scored:'+label+name);graphs[name]=(g,y);nodes[label+':'+name]=guard.graph_kernels(g.raw_cuda_graph())
   assert not fail,('TASK',checks,fail)
   per={n:[] for n in names};orders=[]
   # Each cohort uses a rotated order and its exact reverse; over four
   # cohorts every arm appears in every exposure position exactly twice.
   for cohort in range(4):
    k=(cohort+si)%4;order=names[k:]+names[:k]
    for seq in (order,tuple(reversed(order))):
     orders.append(list(seq))
     for name in seq:per[name].append(replay(graphs[name][0]))
   assert all(min(v)>fixed for v in per.values()),(label,per,fixed)
   samples.append(per);graphs=d=p=None;cleanup()
  rounds[label]=samples;print('PUBLIC_TIMED',label,flush=True)
 result=dict(complete=True,checks=checks,failures=fail,seeds=SEEDS,round_ns=rounds,fixed_ns=fixed,graph_nodes=nodes,elapsed=time.monotonic()-start,ABI={f'{b}x{l}'+('s' if w else 'f'):v for (b,l,w),v in ABI.items()},symmetric='4 rotated forward/reverse cohorts;8 rounds;all arms each position twice',identities={k:hashlib.sha256(BUNDLE[k].encode()).hexdigest() for k in ('source','operator','lee','input')})
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode();assert len(blob)<10800,len(blob)
 return ['META reviewed operatorstack MIT6d9b7765 and LeeApache601c6541v56 vs exactS19; timed real producerABI;8freshseeds8symmetriccold;allroundns/TASK/AA','RESULT_XZ85 '+blob]
