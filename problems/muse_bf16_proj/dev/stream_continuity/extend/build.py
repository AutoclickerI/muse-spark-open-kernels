"""Frozen S18 whole-layer arms: actual owned stages plus explicit coverage controls."""
import ast,base64,hashlib,json,lzma,re
from pathlib import Path
import patch
H=Path(__file__).resolve().parent;R=H.parents[4]
BASE='95df42fe7b3a31e66d60bc769bae15412c952ad77753b7d15b3beea3d89b3af4'
ARMS={'Q':dict(roles=('qkv_gate','qkv3')),'G':{},'O':dict(roles=('o',)),'P':dict(roles=('pool_router',)),'U':dict(roles=('unpooler','unpooler_slice')),'N13':dict(n13=True),'N2':dict(n2=True),'H13':dict(h13=True),'H2':dict(h2=True),'B1':dict(boundaries=('b1',)),'B2':dict(boundaries=('b2',)),'KV':dict(kv=True)}
ARMS['ALL']=dict(roles=('qkv_gate','qkv3','o','pool_router','unpooler','unpooler_slice'),boundaries=('b1','b2'),n13=True,n2=True,h13=True,h2=True,kv=True)
ARMS['KV_LATE']=dict(kv=True,early=False)
ARMS['LATE']=dict(roles=ARMS['ALL']['roles'],kv=True,early=False)
ARMS['EDITABLE']=dict(ARMS['ALL'],native_only=True)

def literal(s,key):return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key)
def replace(s,key,value):
    n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key);return s.replace(ast.get_source_segment(s,n),key+'='+repr(value),1)
def baseline():
    s=(R/'problems/muse_decode_layer/cand_r3_best7_s18_final.py').read_text();n=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='b85decode');s=lzma.decompress(base64.b85decode(ast.literal_eval(n.args[0]))).decode();assert hashlib.sha256(s.encode()).hexdigest()==BASE;return s

def build():
    base=baseline();out={'A':base,'A2':base};meta={}
    patches='\n'.join(ast.get_source_segment(Path(patch.__file__).read_text(),n) for n in ast.parse(Path(patch.__file__).read_text()).body if isinstance(n,ast.FunctionDef))
    for arm,cfg in ARMS.items():
        s=base
        if cfg.get('n13') or cfg.get('n2'):
            for name in ('RVC_NINDEX0_CUDA','RVC_NBASE1_CUDA','RVC_NBASE2_CUDA'):
                s=replace(s,name,patch.ew_patch_nt1(literal(s,name),cfg.get('n13'),cfg.get('n2')))
            if cfg.get('n13'):
                for n in reversed(ast.parse(s).body):
                    if isinstance(n,ast.FunctionDef) and n.name in ('rvc_nindex0_experts','rvc_nbase1_experts','rvc_nbase2_experts'):
                        fn=ast.get_source_segment(s,n);mark="    mod.run(ptrs, m, data['config']['local_start'], stages, grid, torch.cuda.current_stream(dev).cuda_stream, flags)";assert mark in fn
                        new=fn.replace(mark,"    if len(ptrs)==13:ptrs.append(0)\n    ptrs.append(EW_READY.data_ptr() if EW_READY is not None else 0)\n"+mark);s=s.replace(fn,new,1)
        if cfg.get('h13') or cfg.get('h2'):s=replace(s,'P3B_CUDA',patch.ew_patch_heavy(literal(s,'P3B_CUDA'),cfg.get('h13'),cfg.get('h2')))
        if cfg.get('boundaries'):
            s=replace(s,'GLUE_CL_CUDA',patch.ew_patch_glue(literal(s,'GLUE_CL_CUDA'),cfg['boundaries']))
            for n in reversed(ast.parse(s).body):
                if isinstance(n,ast.FunctionDef) and n.name in ('_boundary_fast','_boundary_late'):
                    fn=ast.get_source_segment(s,n);new=patch.ew_patch_glue_triton(fn,cfg['boundaries']);s=s.replace(fn,new,1)
            # Define the helper before importing the modified JIT definitions.
            j=s.index('@triton.jit');s=s[:j]+(H/'glue_triton.py.inc').read_text()+'\n'+s[j:]

        if cfg.get('n13'):
            for name in ('GLUE_CL_CUDA','GP_CUDA'):s=replace(s,name,patch.ew_patch_publish_routes(literal(s,name)))
        if cfg.get('kv'):
            for name in ('PARTIAL_CUDA','PARTIAL_FAST_CUDA','ATTN_PORT_CUDA_SRC'):
                try:old=literal(s,name)
                except (ValueError,StopIteration):continue
                cu,_=patch.ew_patch_attention(old);s=replace(s,name,cu)
        if cfg.get('kv') and not cfg.get('early',True):
            cu=literal(s,'VQ8_CUDA');at=cu.index('DI void producer(');brace=cu.index('{',at)
            cu=cu[:brace+1]+'\n  asm volatile("griddepcontrol.wait;":::"memory");\n'+cu[brace+1:];s=replace(s,'VQ8_CUDA',cu)
        suffix='\nimport re\n'+patches+'\n'

        suffix+='EW_ROLES='+repr(cfg.get('roles',()))+'\nEW_BOUNDARIES='+repr(cfg.get('boundaries',()))+'\nEW_N13='+repr(bool(cfg.get('n13')))+'\nEW_NATIVE_ONLY='+repr(bool(cfg.get('native_only')))+'\nEW_KV='+repr(bool(cfg.get('kv')))+'\nEW_EARLY='+repr(int(cfg.get('early',True)))+'\n'
        if cfg.get('roles') and cfg.get('early',True):
            suffix+='''
_ew_old_port=_port_cuda
_EW_NF_AUDIT=[]
def _port_cuda(src,**kwargs):
    src=_ew_old_port(src,**kwargs)
    if kwargs.get('m32w'):return ew_patch_nf_mw(src)
    if '    auto load = [&](int slot, int s) {' in src:
        src,why=ew_patch_nf_small(src);_EW_NF_AUDIT.append(why)
    return src
_ew_old_wide=_nf_pdl_wide
def _nf_pdl_wide(src):
    src=_ew_old_wide(src)
    if 'auto load=[&](int i)' in src:src,why=ew_patch_nf_wide(src);_EW_NF_AUDIT.append(why)
    return src
'''
        suffix+='EW_DENSE_CUDA='+repr((H/'dense.cu.inc').read_text())+'\n'+(H/'runtime.py.inc').read_text()
        s+=suffix;compile(s,arm,'exec');out[arm]=s;(H/(arm+'_trial.py')).write_text(s)
        meta[arm]=dict(sha256=hashlib.sha256(s.encode()).hexdigest(),config=cfg)
    (H/'manifest.json').write_text(json.dumps(dict(base_sha256=BASE,base='S18',arms=meta,coverage=dict(G='gate already computes its full GEMV before waiting for attention',H13='incoming route fork event cannot overlap; native own-stage placement tested',KV='native histories already early at d1S/v1/v16F/v32F; remaining partial first stage moves; FA3 opaque at v16S/v32S')),indent=2)+'\n')
    return out
if __name__=='__main__':print({a:hashlib.sha256(s.encode()).hexdigest()[:8] for a,s in build().items()})
