"""Compile every changed native transport and the real T128 Triton path for Hopper."""
import importlib.util,hashlib,json,re,subprocess,sysconfig,concurrent.futures
from pathlib import Path
import torch,triton,sys
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent;vector='--vector' in sys.argv;view='--view' in sys.argv;out=H/('native_view' if view else 'native_vector' if vector else 'native');out.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('dc_native',H/('view_candidate.py' if view else 'vector_candidate.py' if vector else 'candidate.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
units=[]
class Mock:
    def __getattr__(self,name):return lambda *a,**kw:None
def capture(name,cpp_sources='',cuda_sources='',**kw):
    units.append((name,cuda_sources));return Mock()
ns=m.dc_nf_vendor();ns['load_inline']=capture
# Instantiate exactly the S15 selected configurations, including cluster paths
# that cannot launch on the local GPU. No Hopper code is run here.
for family,builder,insts in (
 ('d9','v18d_v18d_native_build',[(1,1,1,2,2,0,256,0,1280,8192,1,4,8,4)]),
 ('sm2','tt_v10_sm2_smallm2_build',[(1,1,1,2,0,0,1024,0,8192,1024,4,8,2,1,7)]),
 ('v18','v18s_build',[(1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6)])):
    ns[builder](insts,'dc_compile_'+family)
ns['dd_decodediet_cps_build']('sm',[(1,2,1,2,0,0,256,0,8192,4096,2,4,16,1)],2)
ns['tt_v10_m32x__q_build']()
cpp,src,name=ns['mt_M16TRACE_SOURCES']['o'];units.append(('dc_o16',src))
units.append(('dc_mw',m._port_cuda(ns['dd_decodediet_tt_mw_cuda_source']([(32,64,4,2,2,True,1)]),m32w=True)))
ns['b2o_OLAYER_CUDA']=m._nf_pdl_wide(m.dc_ol_source(ns['b2o_OLAYER_CUDA']));ns['b2o_build_olayer']([(32,128,8,1,1,1)])
units.append(('dc_cd',m.dc_cd_source()))
base=['-std=c++20','-O3','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-DTORCH_EXTENSION_NAME=dc_compile','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']]
base += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
def one(item):
    name,src=item;name=re.sub(r'[^a-zA-Z0-9_]','_',name);f=out/(name+'.cu');f.write_text(src)
    p=subprocess.run(['nvcc',*base,'-c',str(f),'-o',str(f.with_suffix('.o'))],capture_output=True,text=True,timeout=900)
    log=p.stdout+p.stderr;f.with_suffix('.log').write_text(log)
    row=dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),ok=p.returncode==0,
        registers=[int(x) for x in re.findall(r'Used (\d+) registers',log)],
        stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),
        error=log[-2500:] if p.returncode else '')
    print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(3) as pool:rows=list(pool.map(one,units))
# Compile the actual TMA-X / point-decoded-W kernel without launching it.
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
old=driver.active.get_current_target
try:
    driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
    x=torch.empty((128,8192),device='cuda',dtype=torch.bfloat16);w=torch.empty(1,device='cuda',dtype=torch.uint8);p=torch.empty((3,128,2304),device='cuda')
    xd=m._VDSK_TD.from_tensor(x,[128,64])
    k=m._dc_vdsk_gemm.warmup(x,w,p,xd,None,128,2304,8192,3,BT=128,BW=64,BK=64,NS=8,TMA=True,num_warps=4,num_stages=8,grid=(36,3))
    (out/'dc_vdsk.ptx').write_text(k.asm['ptx'])
    rows.append(dict(name='dc_vdsk',ok=True,sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest(),shared=k.metadata.shared,metadata=str(k.metadata)))
finally:driver.active.get_current_target=old
(H/('view_native_compile.json' if view else 'vector_native_compile.json' if vector else 'native_compile.json')).write_text(json.dumps(rows,indent=2)+'\n');assert all(r['ok'] for r in rows)
