from concurrent.futures import ThreadPoolExecutor

def module(name,src):
 f='<moe-public-'+name+'>';linecache.cache[f]=(len(src),None,src.splitlines(True),f);m=types.ModuleType('moe_public_'+name);m.__file__=f;exec(compile(src,f,'exec'),m.__dict__);return m

def lee_module(kind,flags,emulate=False):
 from torch.utils.cpp_extension import load_inline
 arch='90a' if not emulate else ''.join(map(str,torch.cuda.get_device_capability()))
 cu=BUNDLE['lee_'+kind.lower()]
 if emulate:cu=cu.replace('defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900','defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000')
 tag='mp_'+hashlib.sha256((cu+str(flags)+arch).encode()).hexdigest()[:14]
 at=cu.index('#define HD');cu=cu[:at]+'namespace '+tag+' {\n'+cu[at:]+'\n}\nusing '+tag+'::port_'+kind.lower()+';\n'
 # A namespace alias is insufficient for load_inline's separate binding TU:
 # expose a non-namespaced shim with a distinct C++ name.
 fn='port_'+kind.lower();cu=cu.replace('using '+tag+'::'+fn+';', 'void '+fn+'(std::vector<int64_t>a,int64_t m,int64_t lo,'+('int64_t mode,' if kind=='X' else '')+'int64_t st){'+tag+'::'+fn+'(a,m,lo,'+('mode,' if kind=='X' else '')+'st);}')
 cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nvoid '+fn+'(std::vector<int64_t>,int64_t,int64_t,'+('int64_t,' if kind=='X' else '')+'int64_t);'
 return load_inline(name=tag,cpp_sources=cpp,cuda_sources=cu,functions=[fn],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','--use_fast_math','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}']+list(flags))

def s19_prepare(base,w):
 # Only S19's MoE banks are needed. No attention/projection build or tuning.
 dev=w['w13'].device;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 raw=dict(gs13=w['w13_gs'][:32].contiguous(),gs2=w['w2_gs'][:32].contiguous(),pnw=w['post_norm_w'],grid=sms,cnt=torch.zeros(base.NCNT,dtype=torch.int32,device=dev))
 native=base._prepare_recoded_bank(w,raw,None)
 small=dict(gs13=native['gs13'],gs2=native['gs2'],pnw=native['pnw'],sms=sms,banks={'lop3':dict(w13p=native['w13p'],w2p=native['w2p'])})
 for fn in (base.smallm_bind,base.rvc_nfidx0_bind,base.rvc_nbase1_bind,base.rvc_nbase2_bind):fn(small,base.SMALLM_FLAGS)
 heavy=base.nf_perm(base.nf_wide(base.issue_prepare(base.nf_ilv(w))))
 base.NF_OPTS['pmz']=base.p3b_module().pmz_init();base.p3b_module().ch_info(4,0);base.p3b_module().ch_info(4,1)
 return dict(small=small,native=native,heavy=heavy,streams=(torch.cuda.Stream(),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(),torch.cuda.Event()))

def s19_plan(base,data,state):
 M=data['hidden_states'].shape[0]
 if M<=32:
  def fused():return base.smallm_experts(data,state['small'],base.SMALLM_FLAGS[1 if M==1 else 64])
  # Preserve exact allocation/dispatch for whole graph. Phase timing is N/A.
  out=fused()
  return dict(steps=[('fused',fused)],out=out,owners=(data,state))
 p=base.p3b_plan(data,state['heavy'],state['native'],state['streams'])
 h13,h2,nv,ids,tw,pos,tasks,l13,l2=p['owners'];M=data['hidden_states'].shape[0];mod=base.p3b_module();pm=base.mt_pmz_module();sms=torch.cuda.get_device_properties(data['hidden_states'].device).multi_processor_count;stages=2
 while stages>1 and mod.pmz_smem(stages,M)>base.NF_OPTS['pmz']:stages-=1
 grids=[sms*mod.pmz_resident(stages,M,f) for f in (0,1)]
 ptr=lambda ts:[t.data_ptr() for t in ts]
 p13,p2,pn,pl13,pl2=map(ptr,(h13,h2,nv,l13,l2));side,fork,join,light,ljoin=state['streams']
 def phase(first):
  parent=torch.cuda.current_stream();fork.record(parent);side.wait_event(fork);light.wait_event(fork)
  with torch.cuda.stream(side):mod.ch_run(p13 if first else p2,int(first),4,sms*4,M,side.cuda_stream);join.record()
  with torch.cuda.stream(light):mod.br_run(pl13 if first else pl2,70 if first else 71,sms*4,M,light.cuda_stream);ljoin.record()
  (mod if first else pm).pmz_run(pn,int(first),8,M,stages,grids[int(first)],parent.cuda_stream)
  parent.wait_event(join);parent.wait_event(ljoin)
 p['phase_steps']=[('route',p['steps'][0]),('w13',lambda:phase(True)),('w2',lambda:phase(False)),('finish',p['steps'][2])]
 p['steps']=[(k,f) for k,f in zip(('route','body','finish'),p['steps'])]
 return p

def run_plan(p):
 out=None
 for _,fn in p['steps']:out=fn()
 return p['out'] if out is None else out

def capture(fn):
 s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(s):
  for _ in range(2):out=fn()
 torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=fn()
 return g,out

def phase_graphs(p):
 steps=p.get('phase_steps',p['steps']);groups={}
 for label,f in steps:groups.setdefault(label,[]).append(f)
 result={}
 for label,fs in groups.items():
  def call(fs=fs):
   for f in fs:f()
  result[label]=capture(call)[0]
 return result
