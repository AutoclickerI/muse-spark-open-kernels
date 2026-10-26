LAB_RESOURCES={'compile_s':180,'run_s':160,'output_bytes':10000,'context_poison':'none'}
SEEDS=tuple(275163221+104729*i for i in range(8))
def abi_final_module():
 from torch.utils.cpp_extension import load_inline
 arch='90a' if torch.cuda.get_device_capability()==(9,0) else ''.join(map(str,torch.cuda.get_device_capability()))
 cu=BUNDLE['abi_finalize'];tag='mp_abi_'+hashlib.sha256(cu.encode()).hexdigest()[:12]+'_'+arch
 return load_inline(name=tag,cpp_sources='#include <cstdint>\n#include <vector>\n#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\nvoid abi_finish(std::vector<int64_t>,int64_t,int64_t);',cuda_sources=cu,functions=['abi_finish'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','--use_fast_math','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])

def lab(deadline,weights,state=None):
 started=time.monotonic();base=module('S19abi',BUNDLE['base']);op=module('Oabi',BUNDLE['abi_operator']);ck=module('TASKabi',BUNDLE['checker']);final=abi_final_module();checks={};rounds={};phase_rounds={}
 w=ck.generate_weights(275163217);w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in w.items()};bank=op.prepare(w);sb=s19_prepare(base,w)
 def plans(d):
  a=s19_plan(base,d,sb);aa=s19_plan(base,d,sb);o={k:op.abi_plan(d,bank,s19_plan(base,d,sb),v) for k,v in (('O13','w13'),('O2','w2'),('O132','both'))};f=s19_plan(base,d,sb);h13,h2,nv,ids,tw,pos,tasks,l13,l2=f['owners'];fp=[h13[5],ids,tw,pos,h13[11],h13[12],f['out']];ptr=[t.data_ptr() for t in fp]
  def finish():final.abi_finish(ptr,d['hidden_states'].shape[0],torch.cuda.current_stream().cuda_stream);return f['out']
  f['steps']=[(k,finish if k=='finish' else fn) for k,fn in f['steps']];f['phase_steps']=[(k,finish if k=='finish' else fn) for k,fn in f['phase_steps']]
  return dict(A=a,A2=aa,**o,LF=f)
 e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 def timegraph(g):
  lab_cold_l2();e0.record();g.replay();e1.record();e1.synchronize();return round(e0.elapsed_time(e1)*1e6)
 empty=torch.cuda.CUDAGraph()
 with torch.cuda.graph(empty):pass
 fixed=statistics.mean(timegraph(empty) for _ in range(16))
 for M,dist in ((33,'normal'),(257,'normal'),(128,'skewed')):
  d=ck.generate_input(bs=M,sharding='ep8',seed=275163199+M,dist=dist);ref=ck.ref_kernel(d,w);ps=plans(d)
  for n,p in ps.items():
   y=run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);assert ok,('nearby',M,n,msg);g,y=capture(lambda p=p:run_plan(p));g.replay();g.replay();torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);assert ok,('nearby graph',M,n,msg);checks[n]=checks.get(n,0)+2;del g
  ps=d=ref=None;gc.collect();torch.cuda.empty_cache()
 for M in (128,256):
  samples=[];phases=[]
  for si,seed in enumerate(SEEDS):
   assert time.monotonic()<deadline-8,('deadline',M,seed)
   d=ck.generate_input(bs=M,sharding='ep8',seed=seed);ref=ck.ref_kernel(d,w);ps=plans(d);gs={}
   for n,p in ps.items():
    g,y=capture(lambda p=p:run_plan(p));g.replay();torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);assert ok,('TASK',M,seed,n,msg);checks[n]=checks.get(n,0)+1;gs[n]=g
   names=tuple(ps);per={n:[] for n in names}
   for c in range(2):
    k=(si+c)%len(names);order=names[k:]+names[:k]
    for seq in (order,tuple(reversed(order))):
     for n in seq:per[n].append(timegraph(gs[n]))
   samples.append(per)
   selected={n:ps[n] for n in ('A','O132','LF')};pgs={};pv={}
   for n,p in selected.items():
    run_plan(p);pgs[n]=phase_graphs(p)
    for stage in ('route','w13','w2','finish'):
     if n=='LF' and stage!='finish':continue
     pv[n+':'+stage]=[]
   keys=tuple(pv)
   for c in range(2):
    k=(si+c)%len(keys);order=keys[k:]+keys[:k]
    for seq in (order,tuple(reversed(order))):
     for key in seq:
      n,stage=key.split(':')
      for label,fn in selected[n]['phase_steps']:
       if label==stage:break
       fn()
      pv[key].append(timegraph(pgs[n][stage]))
   phases.append(pv);ps=gs=pgs=selected=d=ref=None;gc.collect();torch.cuda.empty_cache()
  rounds[str(M)]=samples;phase_rounds[str(M)]=phases
 result=dict(complete=True,seeds=SEEDS,weight_seed=275163217,checks=checks,failures=[],fixed_ns=fixed,round_ns=rounds,phase_ns=phase_rounds,identities={k:hashlib.sha256(v.encode()).hexdigest() for k,v in BUNDLE.items()},elapsed_s=time.monotonic()-started,protocol='S19 sorted BF16 inter, FP32 Y and64FP32 SS tiles; all conversion/metadata work timed; 8freshseeds4symmetriccold; isolated phases and unchanged whole control')
 blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode(),preset=9)).decode();assert len(blob)<9300,len(blob)
 return ['ABI_RESULT_XZ85 '+blob]
