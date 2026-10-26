"""Compile transformed native code for90a and keep exact machine-code placement evidence."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,subprocess,sys,sysconfig
from pathlib import Path
from torch.utils.cpp_extension import include_paths
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'problems/muse_decode_layer'))
sources={}
for arm,names in {'N13':('RVC_NINDEX0_CUDA','RVC_NBASE1_CUDA','RVC_NBASE2_CUDA'),'N2':('RVC_NBASE2_CUDA',),'H13':('P3B_CUDA',),'H2':('P3B_CUDA',),'B1':('GLUE_CL_CUDA',),'KV':('PARTIAL_CUDA',),'ALL':('P3B_CUDA','RVC_NBASE2_CUDA','GLUE_CL_CUDA','GP_CUDA')}.items():
 s=(H/(arm+'_trial.py')).read_text()
 for n in ast.parse(s).body:
  if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in names:sources[arm+'_'+n.targets[0].id]=ast.literal_eval(n.value)
sources['dense']=(H/'dense.cu.inc').read_text()
sp=importlib.util.spec_from_file_location('ew_all_native',H/'ALL_trial.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m);ns=m._nf_vendor()
for t,name in ((1,'o'),(8,'o'),(1,'qkv3'),(16,'unpooler'),(32,'qkv_gate')):
 fam,cfg,mt,st,red,ntmax,extra=m.NF_SMALL[t,name];n,k=(8192,1024) if name=='o' else (8192,4096) if name=='unpooler' else (1280,8192) if name=='qkv3' else (2304,8192)
 inst=(1,mt,1,st,red,0,ntmax,0,n,k,*cfg,*extra)
 gen={'sm2':'tt_v10_sm2_smallm2_cuda_source','v18':'v18s_cuda_source','d9':'v18d_v18d_native_cuda_source','cp':'dd_decodediet_cps_cuda_source'}
 if fam in gen and gen[fam] in ns:
  fn=ns[gen[fam]]
  try:cu=fn([inst])
  except TypeError:cu=fn('sm',[inst],2)
  sources['NF_'+str(t)+'_'+name]=cu
 elif fam=='cp':
  def capture(*args,**kw):sources['NF_'+str(t)+'_'+name]=kw['cuda_sources'];return None
  old=ns['load_inline'];ns['load_inline']=capture
  ns['dd_decodediet_cps_build']('sm',[inst],2);ns['load_inline']=old
 elif fam=='q32':sources['NF_'+str(t)+'_'+name]=ns['tt_v10_m32x__Q_CUDA']
# Exact wide source is instantiated by its vendor builder. Intercept only the
# build call to retain its emitted sources; no GPU calculation or module edit.
for key,gen,cfg in (('b2o_OLAYER_CUDA','b2o_olayer_source',(32,128,8,1,1,1)),('OLAYER_CUDA','olayer_source',(128,128,4,3,1,1))):
 ns[key]=m._nf_pdl_wide(ns[key]);sources['NF_'+key]=ns[gen]([cfg])
sources['NF_MW32']=m._port_cuda(ns['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1),(64,64,4,2,2,True,1)]),m32w=True)
sources['NF_16_o']=ns['mt_M16TRACE_SOURCES']['o'][1]
sources['P10_pool']=ns['v18s_cuda_source']([(1,mt,1,1,0,0,256,0,4608,8192,1,8,16,1,6) for mt in (1,2)])
base=(H/'out/base.py').read_text()
for n in ast.parse(base).body:
 if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('GLUE_CL_CUDA','P3B_CUDA','PARTIAL_CUDA','RVC_NINDEX0_CUDA','RVC_NBASE1_CUDA','RVC_NBASE2_CUDA'):sources['BASE_'+n.targets[0].id]=ast.literal_eval(n.value)
sp=importlib.util.spec_from_file_location('ew_base_native',H/'out/base.py');bm=importlib.util.module_from_spec(sp);sys.modules[sp.name]=bm;sp.loader.exec_module(bm);bn=bm._nf_vendor()
for t,name in ((1,'o'),(8,'o'),(1,'qkv3'),(16,'unpooler')):
 fam,cfg,mt,st,red,ntmax,extra=bm.NF_SMALL[t,name];n,k=(8192,1024) if name=='o' else (8192,4096) if name=='unpooler' else (1280,8192)
 inst=(1,mt,1,st,red,0,ntmax,0,n,k,*cfg,*extra)
 if fam=='cp':
  def capture(*args,**kw):sources['BASE_NF_'+str(t)+'_'+name]=kw['cuda_sources'];return None
  old=bn['load_inline'];bn['load_inline']=capture;bn['dd_decodediet_cps_build']('sm',[inst],2);bn['load_inline']=old
 else:sources['BASE_NF_'+str(t)+'_'+name]=bn[{'sm2':'tt_v10_sm2_smallm2_cuda_source','v18':'v18s_cuda_source','d9':'v18d_v18d_native_cuda_source'}[fam]]([inst])
sources['BASE_NF_32_qkv_gate']=bn['tt_v10_m32x__Q_CUDA']
for key,gen,cfg in (('b2o_OLAYER_CUDA','b2o_olayer_source',(32,128,8,1,1,1)),('OLAYER_CUDA','olayer_source',(128,128,4,3,1,1))):
 bn[key]=bm._nf_pdl_wide(bn[key]);sources['BASE_NF_'+key]=bn[gen]([cfg])
sources['BASE_NF_MW32']=bm._port_cuda(bn['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1),(64,64,4,2,2,True,1)]),m32w=True)
sources['BASE_NF_16_o']=bn['mt_M16TRACE_SOURCES']['o'][1]
sources['BASE_P10_pool']=bn['v18s_cuda_source']([(1,mt,1,1,0,0,256,0,4608,8192,1,8,16,1,6) for mt in (1,2)])

out=H/'native';out.mkdir(exist_ok=True)
flags=['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-DTORCH_EXTENSION_NAME=ew_native','-isystem',sysconfig.get_paths()['include']]
for p in include_paths(device_type='cuda'):flags+=['-isystem',p]
def one(item):
 name,src=item;p=out/(name+'.cu');p.write_text(src);res=subprocess.run(flags+['-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=res.stdout+res.stderr;p.with_suffix('.log').write_text(log)
 row=dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),ok=res.returncode==0,errors=[l for l in log.splitlines() if 'error' in l][:6])
 if row['ok']:
  sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
  row['registers']=[int(v) for v in re.findall(r'Used (\d+) registers',log)];row['spills']=bool(re.search(r'[1-9][0-9]* bytes spill',log));row['acquires']=sass.count('ACQBULK');row['async_copies']=sass.count('LDGSTS')+sass.count('UTMALDG')
 return row
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,sources.items()))
(H/'native_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r,flush=True)
assert all(r['ok'] for r in rows)
