"""S20 on the exact S19 expansion; every new recipe has an independent leave-out."""
import ast, base64, hashlib, importlib.util, json, lzma, sys
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path.insert(0,str(H.parent/'s16'))
import patches
def load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s19=load(H.parent/'s19/compose.py','s20_s19_compose')
sha=s19.sha
BASE_SHA='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a'
ATT=R/'problems/muse_attention_decode/dev/full_attention'
DA=R/'problems/muse_bf16_proj/dev/dense_all'
EW=H/'inputs'
IP=H.parent/'nt1_timeline/structural/intpipe'
DEFAULT=('C3','GATE','M64','PRIVATE1','AFFINE32','H2','B1')
ROWS={'C3':['d16F','d32F'],'GATE':['d1S'],'M64':['v1F'],
      'PRIVATE1':['d1S','d1F'],'AFFINE32':['d32S','d32F'],
      'H2':['v16S','v16F','v32S','v32F'],
      'B1':['d1S','d1F','d16S','d16F','d32S','d32F','v1S','v1F','v16S','v16F','v32S','v32F']}

M64='''
_S20_META64=None
def _s20_m64_attention(qkv,data,state):
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    c=data['config'];b=c['batch'];q,gate=qk_prep(qkv,data,state,q_fp8=True)
    descale={} if c['kv_scale']==1 else dict(k_descale=data['k_scale'].reshape(1,1).expand(b,HKV),v_descale=data['v_scale'].reshape(1,1).expand(b,HKV))
    a=flash_attn_with_kvcache(q=q,k_cache=data['k_cache'],v_cache=data['v_cache'],page_table=data['page_table'],cache_seqlens=data['cache_seqlens'],cu_seqlens_q=data['cu_seqlens_q'],max_seqlen_q=c['q_len'],softmax_scale=c['sm_scale']*c['qk_scale_factor'],causal=True,window_size=(-1,-1),scheduler_metadata=_S20_META64,num_splits=64,**descale)
    if isinstance(a,tuple):a=a[0]
    out=torch.empty((b*c['q_len'],Q_SIZE),device=q.device,dtype=torch.bfloat16)
    _launch(_norm_gate_kernel,(b*c['q_len'],),a,gate,out,RMS_EPS,HQ=HQ,D=HD,num_warps=4)
    return out
_s20_before_m64=_best7_attention
def _best7_attention(qkv,data,state,plan):
    c=data['config']
    if (c['batch'],c['q_len'],c['window'])==(1,8,0):return _s20_m64_attention(qkv,data,state)
    return _s20_before_m64(qkv,data,state,plan)
'''

B1_DISPATCH='''
_s20_before_b1=_best7_boundary
def _best7_boundary(branch,stream,state,plan,phase,sync=None):
    if phase!='b1' or 'rt_workspace' in plan:return _s20_before_b1(branch,stream,state,plan,phase,sync)
    geo=plan.get('geometry',{}).get(phase,())
    if geo and geo[0]=='cl':
        return _glue_cl_boundary(branch,stream,state['alpha1'],state['beta1'],state['pre_ffn_w'],None,geo)
    return _s20_before_b1(branch,stream,state,plan,phase,sync)
'''

def literal(source,name):return ast.literal_eval(patches.bindings(source)[name][0].value)
def assign(source,name,value):return s19.s18c.assign(source,name,value)
def b1_triton(fn,ew):
    if '    alpha=_pa_f32(ALPHA+j)' in fn:return ew.ew_patch_glue_triton(fn,('b1',))
    # S19 reverted S18's late-vector hoist. Keep that exact b2 path; stage
    # immutable b1 vectors in the first slot as in the nominated recipe.
    mark='    _pdl_wait(USE_PDL)\n';assert fn.count(mark)==1
    head,tail=fn.split(mark)
    staged=tail.replace('    j = tl.arange(0, 8192)','    j = tl.arange(0, 8192)\n    ew_idx=tl.minimum(j,127)')
    for name,v in [('ALPHA','ew_a'),('BETA','ew_b'),('PREW','ew_c')]:
        old=f'tl.load({name} + j)';assert old in staged
        staged=staged.replace(old,f'tl.where(j<128,tl.gather({v},ew_idx,0),tl.load({name}+j,j>=128,0.))')
    staged='    ew_a,ew_b,ew_c,ew_d=_ew_immutable_stage(ALPHA,BETA,PREW,ALPHA)\n'+staged
    indent=lambda s:'\n'.join('    '+line for line in s.rstrip().splitlines())+'\n'
    return head+'    if not HAS_POSTW:\n'+indent(staged)+'    else:\n'+indent(mark+tail)
def expand(path):
    packed=path.read_text()
    call=next(n for n in ast.walk(ast.parse(packed)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode')
    return lzma.decompress(base64.b85decode(ast.literal_eval(call.args[0]))).decode()
def base():
    text=expand(P/'cand_r3_best7_s19_final.py');assert sha(text)==BASE_SHA;return text

def build(enabled=DEFAULT):
    enabled=set(enabled);assert enabled<=set(DEFAULT)
    source=base()
    # Keep one last warm step, after all new banks, in both the host and every arm.
    warm="_s19_prior_warm=_s16_prepare_delta\ndef _s16_prepare_delta(weights,state):\n    state=_s19_prior_warm(weights,state)\n    import time as _t\n    _t0=_t.monotonic()\n    _s19_warm(state,weights['qkv_gate'].device)\n    state.setdefault('s19_prepare_s',{})['warm']=_t.monotonic()-_t0\n    return state\n"
    source=s19.swap(source,warm,'')
    if 'C3' in enabled:
        cu=(H/'inputs/C3.cu').read_text();assert sha(cu)=='519ea55f1acf936ac115e6e42b5d0c7abc6b913418865ebd2fe54d2d49932208'
        source=assign(source,'ATTN_PORT_CUDA_SRC',cu)
    if 'PRIVATE1' in enabled:
        transform=load(IP/'build.py','s20_address').transform
        source=assign(source,'RVC_NFIDX0_CUDA',transform(literal(source,'RVC_NFIDX0_CUDA'),'PRIVATE'))
    if 'AFFINE32' in enabled:
        cell=(IP/'affine2_cell.py').read_text().replace('ipc_affine2','rvc_s20aff32').replace('IPC_AFFINE2','RVC_S20AFF32')
        source+='\n'+cell+'''
_s16_native_builds.append(rvc_s20aff32_module)
_s20_before_affine32=smallm_experts
def smallm_experts(data,state,flags,stages=2,grid=264):
    if data['hidden_states'].shape[0] in (32,):return rvc_s20aff32_experts(data,state,flags,stages,grid)
    return _s20_before_affine32(data,state,flags,stages,grid)
'''
        source=s19.delta(source,'AFFINE32',"    rvc_s20aff32_bind(state['s4_nt1_bank'],{32:69})")
    ew=load(EW/'patch.py','s20_early_weights')
    if 'H2' in enabled:source=assign(source,'P3B_CUDA',ew.ew_patch_heavy(literal(source,'P3B_CUDA'),w2=True))
    if 'B1' in enabled:
        source=assign(source,'GLUE_CL_CUDA',ew.ew_patch_glue(literal(source,'GLUE_CL_CUDA'),('b1',)))
        for name in ('_boundary_fast','_boundary_late'):
            old=patches.segment(source,patches.bindings(source)[name][0]);new=b1_triton(old,ew);source=s19.swap(source,old,new)
        at=source.index('@triton.jit');source=source[:at]+(EW/'glue_triton.py.inc').read_text()+'\n'+source[at:]
        source+=B1_DISPATCH
    if 'GATE' in enabled:
        gate=load(DA/'patch_gate.py','s20_gate');source=gate.apply(source)
        source=s19.swap(source,'_da_layer_prepare=prepare\n','')
        wrapper="def prepare(weights):\n    state=_da_layer_prepare(weights)\n    state['da']=da_prepare(weights,DA_LOCAL_REDUCE,DA_WIDE_STANDIN)\n    state['da_prepare_s']=dict(state['da']['times'])\n    return state\n"
        source=s19.swap(source,wrapper,'')
        source=s19.delta(source,'GATE',"    state['da']=da_prepare(weights,DA_LOCAL_REDUCE,DA_WIDE_STANDIN)\n    state['da_prepare_s']=dict(state['da']['times'])")
    if 'M64' in enabled:
        source+=M64
        source=s19.delta(source,'M64',"    global _S20_META64\n    _S20_META64=torch.tensor([64,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],device=weights['qkv_gate'].device,dtype=torch.int32)")
    source=s19.delta(source,'warm',s19.WARM)
    compile(source,'<S20>','exec')
    return source,dict(sha256=sha(source),base_sha256=BASE_SHA,enabled=[x for x in DEFAULT if x in enabled])

if __name__=='__main__':
    (H/'generated').mkdir(exist_ok=True)
    s,m=build();(H/'generated/s20.py').write_text(s);(H/'selection.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
