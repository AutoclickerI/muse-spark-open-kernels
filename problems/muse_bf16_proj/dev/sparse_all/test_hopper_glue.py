"""Force the actual native plan builders and launch ABIs with arithmetic stand-ins."""
import importlib.util,math,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('dc_glue',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(61791)
def dense(n,k):return (torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
w=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),router=.045*torch.randn((256,8192),device='cuda',generator=g),unpooler=dense(8192,4096))
roles=m.dc_role_weights(w);calls=[];registry={};current={}
class Standin:
    def __getattr__(self,label):
        def run(*args):
            if not current:return 0
            role=current['role'];x=current['x'];plan=current.get('plan');t,k=x.shape;n=roles[role].shape[0];stream=torch.cuda.current_stream().cuda_stream
            if role=='o':
                assert args[:3]==(0,x.data_ptr(),current['bank']['coded'].data_ptr()) and args[4:]==(t,stream,0),args
                out=registry[args[3]];decoded=m.dc_unpack(current['bank'],torch.empty_like(roles[role]));out.copy_(torch.mm(x,decoded.T))
            elif plan.get('kind')=='mw':
                assert args==(plan['idx'],x.data_ptr(),plan['wp'].data_ptr(),plan['esc'].data_ptr(),args[4],t,n,k,1,0,stream),args
                registry[args[4]].copy_(torch.mm(x,roles[role].T,out_dtype=torch.float32))
            else:
                assert args==(plan['idx'],plan['wp'].data_ptr(),x.data_ptr(),args[3],plan['esc'].data_ptr(),t,n,k,*plan['cfg'],plan['t01'],plan['u01'],stream,0),args
                registry[args[3]].copy_(torch.mm(x,roles[role].T,out_dtype=torch.float32))
            calls.append(dict(role=role,t=t,entry=label,captured=torch.cuda.is_current_stream_capturing()));return 0
        return run
def mock(*a,**kw):
    source=kw.get('cuda_sources','')
    if 'r3_o32_o_direct_a_ol_projection' in source:
        assert 'const __grid_constant__ CUtensorMap tw,' not in source
        assert 'CUtensorMap tw{}' not in source
        assert '>>>(tx,' in source
        assert 'map(&tw' not in source
    return Standin()
old_vendor=m._dc_base_nf_vendor;old_loader=m.load_inline;old_cap=torch.cuda.get_device_capability
def vendor():
    ns=old_vendor();ns['load_inline']=mock;return ns
try:
    m._dc_base_nf_vendor=vendor;m.load_inline=mock;torch.cuda.get_device_capability=lambda *a,**kw:(9,0)
    keys={(1,'qkv3'),(32,'qkv3'),(32,'qkv_gate'),(32,'unpooler')}
    bank=m.dc_nf_prepare(w,keys);assert set(bank['plans'])==keys
    oc,_=m.dc_encode_cd(m.cd_encode(w['o']));st=dict(native=True,roles=roles,nf_sparse=bank,sparse={'o':oc})
    m.dc_prepare_ol(st)
finally:m._dc_base_nf_vendor=old_vendor;m.load_inline=old_loader;torch.cuda.get_device_capability=old_cap
original_empty=torch.empty
def empty(*a,**kw):
    out=original_empty(*a,**kw);registry[out.data_ptr()]=out;return out
torch.empty=empty;rows=[]
try:
    for t,role in sorted(keys|{(32,'o')}):
        x=torch.randn((t,roles[role].shape[1]),device='cuda',generator=g).bfloat16();current.update(role=role,x=x)
        if role=='o':current['bank']=oc
        else:current['plan']=bank['plans'][t,role]
        call=lambda:m.dc_ol_project(x,st) if role=='o' else m.dc_nf_project(x,role,st)
        call();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):p,s=call()
        for seed in (61791,14357,92003):
            g.manual_seed(seed);x.copy_(torch.randn(x.shape,device='cuda',generator=g).bfloat16());graph.replay()
            got=p.double().sum(0);ref=x.double()@roles[role].double().T;err=float((got-ref).norm()/ref.norm())
            assert err<(.004 if p.dtype==torch.bfloat16 else 3e-5),(role,t,err)
            rows.append(dict(role=role,t=t,seed=seed,error=err));print('PASS',rows[-1],flush=True)
finally:torch.empty=original_empty
(H/'hopper_glue_validation.json').write_text(json.dumps(dict(passed=True,candidate_sha256=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest(),rows=rows,calls=calls),indent=2)+'\n')
