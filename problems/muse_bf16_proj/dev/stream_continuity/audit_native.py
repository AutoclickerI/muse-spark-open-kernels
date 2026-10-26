"""Compile every changed selected MoE body and prefetch helper for the real Hopper target."""
import ast,importlib.util,concurrent.futures,hashlib,json,subprocess,sys,sysconfig
from pathlib import Path
from torch.utils.cpp_extension import include_paths
H=Path(__file__).resolve().parent
sources={}
for arm in ('E','P'):
    tree=ast.parse((H/(arm+'_trial.py')).read_text())
    for n in tree.body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('SMALLM_VAR_CUDA','_s16_epi_NORM_EPIN_CUDA','_s16_epi_PAIR_EPIT_CUDA'):
            sources[arm+'_'+n.targets[0].id]=ast.literal_eval(n.value)
sources['prefetch']=(H/'prefetch.cu.inc').read_text()
sys.path.insert(0,str(H.parents[3]/'problems/muse_decode_layer'))
sp=importlib.util.spec_from_file_location('sc_native_W',H/'W_trial.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
ns=m._nf_vendor()
for key in ((1,'o'),(8,'o')):
    family,cfg,mt,st,red,ntmax,extra=m.NF_SMALL[key]
    inst=(1,mt,1,st,red,0,ntmax,0,8192,1024,*cfg,*extra)
    gen={'sm2':'tt_v10_sm2_smallm2_cuda_source','v18':'v18s_cuda_source'}[family]
    sources['W_NF_O'+str(key[0])]=ns[gen]([inst])
    assert 'auto loadw' in sources['W_NF_O'+str(key[0])]
assert m._SC_PORT_AUDIT
out=H/'native';out.mkdir(exist_ok=True)
flags=['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-DTORCH_EXTENSION_NAME=sc_native','-isystem',sysconfig.get_paths()['include']]
for p in include_paths(device_type='cuda'):flags+=['-isystem',p]
def one(item):
    name,src=item;p=out/(name+'.cu');p.write_text(src)
    r=subprocess.run(flags+['-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True)
    p.with_suffix('.log').write_text(r.stdout+r.stderr)
    return dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),ok=r.returncode==0,errors=[l for l in (r.stdout+r.stderr).splitlines() if 'error' in l][:8])
with concurrent.futures.ThreadPoolExecutor(4) as pool:rows=list(pool.map(one,sources.items()))
(H/'native_validation.json').write_text(json.dumps(dict(target='sm_90a',sources=rows),indent=2)+'\n')
for r in rows:print(r)
assert all(r['ok'] for r in rows)
