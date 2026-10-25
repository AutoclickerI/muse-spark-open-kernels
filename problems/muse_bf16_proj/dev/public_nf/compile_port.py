"""Materialize only ported native CUDA units for independent sm90a compile proof."""
import hashlib,json,subprocess,sys,concurrent.futures
from pathlib import Path
import torch
import port_body as p
H=Path(__file__).resolve().parent
original='--original' in sys.argv
ns={}
if original:exec(compile(p.NF_VENDOR,'<audited-NF86aff75-original>','exec'),ns)
else:ns=p._nf_vendor()
units=[]
def capture(name,cpp_sources='',cuda_sources='',**kw):
 if cuda_sources:units.append((name,cuda_sources,kw.get('extra_cuda_cflags',[])))
 return None
ns['load_inline']=capture;ns['_nf_arch_flag']=lambda:'-gencode=arch=compute_90a,code=sm_90a'
grouped={}
for key,(fam,cfg,mt,st,red,ntmax,extra) in p.NF_SMALL.items():
 if original and key[1]=='qkv3':continue
 n,k={'qkv_gate':(2304,8192),'qkv3':(1280,8192),'pool_router':(4608,8192),'o':(8192,1024),'unpooler_slice':(1024,4096),'unpooler':(8192,4096)}[key[1]]
 if original and key[1]=='pool_router':n=4096
 grouped.setdefault(fam,[]).append((1,mt,1,st,red,0,ntmax,0,n,k,*cfg,*extra))
for family,insts in grouped.items():
 insts=list(dict.fromkeys(insts))
 if family=='d9':ns['v18d_v18d_native_build'](insts,'d9')
 elif family=='v18':ns['v18s_build'](insts,'v18')
 elif family=='sm2':ns['tt_v10_sm2_smallm2_build'](insts,'sm2')
 elif family in ('u1','u8'):ns[('tt_v12_smtrace1' if family=='u1' else 'tt_v13_smtrace8')+'_smtrace_build'](insts,family)
 elif family=='cp':ns['dd_decodediet_cps_build']('sm',insts,2)
 elif family in ('q16','q32','p32'):ns[{'q16':'b2q_tt_v10_m32x__q_build','q32':'tt_v10_m32x__q_build','p32':'b2p_tt_v10_m32x__build'}[family]]()
 elif family=='q8':units.append(('q8',ns['BUNDLE_QKV_CUDA'],[]))
 elif family=='o16':units.append(('o16',ns['mt_M16TRACE_SOURCES']['o'][1],[]))
mw=ns['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1),(64,64,4,2,2,True,1)])
units.append(('mw',mw if original else p._port_cuda(mw,m32w=True),[]))
for name in ('OLAYER_CUDA','b2o_OLAYER_CUDA','tt_v10_tune_RL_CUDA','tt_v14_qkv_QKVBIG_CUDA'):ns[name]=ns[name] if original else p._nf_pdl_wide(ns[name])
ns['b2o_build_olayer']([(32,128,8,1,1,1)])
ns['build_olayer']([(128,128,4,3,1,1)])
ns['tt_v10_tune_build_rl']([(256,64,5,7,1,1,1)])
ns['tt_v14_qkv_build_qkv']([(64,36,128,3,8,539,5)])
out=H/('native_original' if original else 'native');out.mkdir(exist_ok=True)
def build(item):
 name,src,flags=item;tag=hashlib.sha256(src.encode()).hexdigest();file=out/(name+'.cu');file.write_text(src)
 cmd=['nvcc','-std=c++20','-O3','-arch=sm_90a','--cubin','-Xptxas=-v',str(file),'-o',str(file.with_suffix('.cubin'))]
 r=subprocess.run(cmd,capture_output=True,text=True)
 file.with_suffix('.log').write_text(r.stdout+r.stderr)
 row=dict(name=name,sha256=tag,ok=r.returncode==0,log=(r.stdout+r.stderr)[-1200:] if r.returncode else '')
 print(name,row['ok'],row['log'],flush=True);return row
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(build,units))
(H/('native_original_compile.json' if original else 'native_compile.json')).write_text(json.dumps(rows,indent=2)+'\n')
assert all(x['ok'] for x in rows),rows
