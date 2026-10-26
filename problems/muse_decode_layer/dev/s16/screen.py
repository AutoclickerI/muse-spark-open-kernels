"""Serial normal/forced local_eval on every reset-arm production patch."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval
import static_cache
static_cache.install(local_eval)


def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def install(module,forced=False):
    import types
    m=module
    if forced:load(H.parent/'best7/force.py','s16_best7_force').install(m,False,False)
    if hasattr(m,'LIBRARY_NATIVE_ROWS'):
        sys.path.insert(0,str(H.parent/'s15'))
        load(H.parent/'library_chain/library_standins.py','s16_library_standins').install(m)
    else:load(H.parent/'s15/local_standins.py','s16_s15_standins').install(m)
    if hasattr(m,'r2_attention'):
        original_r2=m.r2_attention
        def r2_attention(data,qkv=None,kind='last',qt=1,ns=8):
            return original_r2(data,qkv,kind,qt,1 if kind in ('dsm','dsm_old') else ns)
        m.r2_attention=r2_attention
        print('S16_COOP_STANDIN native C1 launch; original QT/math/ABI',flush=True)
    counts={};stubs={}
    # Every new attention driver runs its Python allocation/launch path. Only
    # unavailable native global/DSM kernels use a pointer-level launch stand-in.
    for name,fn in list(vars(m).items()):
        if not name.startswith('_s16_') or not name.endswith(('cs_attention','cc_attention','attention')) or not callable(fn):continue
        if name.endswith('base_attention'):continue
        if name.startswith('_s16_soft_'):
            prefix=name[:-len('attention')];setattr(m,prefix+'module',m._vq8_module)
        elif 'early_' not in name:
            prefix=name[:-len('cs_attention' if name.endswith('cs_attention') else 'cc_attention')]
            modname=prefix+('cs_module' if name.endswith('cs_attention') else 'cc_module')
            context={};original=fn
            def native(*args,_context=context):
                assert args[10]==_context['out'].data_ptr()
                # copy_ is graph-capturable and writes the exact allocated output.
                _context['out'].copy_(_context['reference'].view_as(_context['out']))
            # Capture the allocated output without changing the driver's allocation.
            def stubbed(data,qkv=None,_original=original,_context=context,**kwargs):
                reference=m.attention_partial_port(qkv,data,False)
                _context['reference']=reference
                empty=m.torch.empty
                def allocate(*a,**kw):
                    t=empty(*a,**kw)
                    if t.dtype==m.torch.bfloat16 and tuple(t.shape)==(data['config']['batch'],data['config']['q_len'],m.Q_SIZE):_context['out']=t
                    return t
                # Do not patch torch globally; intercept the driver's global torch
                # binding through a proxy used only for this native-unavailable ABI.
                g=_original.__globals__;old=g['torch'];proxy=types.SimpleNamespace(**vars(old));proxy.empty=allocate
                g['torch']=proxy
                try:return _original(data,qkv,**kwargs)
                finally:g['torch']=old
            setattr(m,modname,lambda _native=native:types.SimpleNamespace(sliding_forward=_native,sliding_resources=lambda *a:[0,0,0,1,1]))
            setattr(m,name,stubbed);stubs[name]='partial-port arithmetic, exact allocation and pointer ABI'
        original=getattr(m,name)
        def counted(*args,_fn=original,_name=name,**kw):
            key=_name;row=counts.setdefault(key,dict(eager=0,captured=0));row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
            return _fn(*args,**kw)
        setattr(m,name,counted)
    # Record new small/heavy/PDL calls without changing their arithmetic.
    for name in ('smallm_experts','nf_call','_best7_boundary','_best7_route'):
        fn=getattr(m,name)
        def counted(*args,_fn=fn,_name=name,**kw):
            row=counts.setdefault(_name,dict(eager=0,captured=0));row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
            return _fn(*args,**kw)
        setattr(m,name,counted)
    if hasattr(m,'_s16_native_builds'):
        m._s16_native_builds=[getattr(m,fn.__name__,fn) for fn in m._s16_native_builds]
    print('S16_LOCAL_STANDINS',stubs,flush=True)
    return counts


def inventory(m,state):
    import types
    rows={};seen=set()
    def walk(path,value):
        if isinstance(value,m.torch.Tensor):
            row=dict(dtype=str(value.dtype),shape=list(value.shape),bytes=value.numel()*value.element_size())
            expert=any(x in path for x in ('w13','w2')) and 'roles.' not in path
            if value.device.type=='cpu':row['holds']='CPU launch configuration; no GPU weight values'
            elif expert:
                last=path.rsplit('.',1)[-1]
                if last.endswith(('_gs','gs13','gs2','gs13_gate','gs13_up','weight_scale_2')):row['holds']='original FP32 expert global scale'
                elif last=='pnw':row['holds']='original BF16 norm'
                else:
                    assert value.dtype in (m.torch.uint8,m.torch.int32,m.torch.float8_e4m3fn),(path,row)
                    row['holds']='packed E2M1/E4M3 codes; no widened expert values'
            elif 'coded_dense' in path or '.nf_port.plans.' in path:row['holds']='lossless BF16 code/layout, original-bit escape metadata, or launch scratch'
            else:row['holds']='original precision dense/norm values, indices or runtime scratch'
            rows[path]=row;return
        if id(value) in seen:return
        seen.add(id(value))
        if isinstance(value,dict):
            for k,v in value.items():walk(path+'.'+str(k),v)
        elif isinstance(value,(tuple,list)):
            for i,v in enumerate(value):walk(path+f'[{i}]',v)
        elif isinstance(value,types.FunctionType) and value.__closure__:
            for i,c in enumerate(value.__closure__):walk(path+f'.closure[{i}]',c.cell_contents)
        elif hasattr(value,'__dict__') and not isinstance(value,(types.ModuleType,type)):
            for k,v in vars(value).items():walk(path+'.'+k,v)
    walk('state',state)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--forced',action='store_true');ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);original=local_eval.load_module;records={}
    def loader(name,path):
        m=original(name,path)
        if name=='submission':
            records['calls']=install(m,a.forced)
            prepared=m.prepare
            def prepare(weights):
                state=prepared(weights)
                (a.out/'precision_inventory.json').write_text(json.dumps(inventory(m,state),indent=2)+'\n')
                return state
            m.prepare=prepare
        return m
    local_eval.load_module=loader
    sys.argv=['local_eval.py',str(a.candidate),'--no-baseline','--reps','4','--inputs','3','--json',str(a.out/'local.json')]
    try:local_eval.main()
    finally:(a.out/'calls.json').write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__':main()
