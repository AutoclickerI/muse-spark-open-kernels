"""S19 on frozen S18: the lanes' handed-off winners, each a removable piece for its leave-out arm.

Pieces (the paired held-out lab confirms the whole layer; 7c0fd3cc chose the rows):
  P10   exact 12-bit QKV at (T,N) (8,2304) only (v1); at T=16 it cost d16 1.2%   2b45988c / 7c0fd3cc
  RAW2  two-WG sliding attention at v32S (PIPE2 is the alternative) 47e1f9d8 / af563b6b
  R8    page-coordinate lookahead in the q1 port kernel (runs only at d16F/d32F) 4df30d04
  D     tensor-core softmax denominator in VQ8 (full verify only)   4df30d04
  FAST  P3a FAST cut merged into S18's SHARD2+INDEX M1 cell         d7fb2d8b
  LATE  pdl_M's _boundary_late edit reverted (T=256 only: v32S/v32F) 7c0fd3cc v32S_no_pdl_M
Alternatives (arms only): P10 at T=16; pdl_M's _boundary_late.
Every candidate and arm is baked (bake()): prepare keeps the compiles, in one parallel pool, and the weight coding;
the H200-measured picks are constants and the private FP32 oracles are gone.
State work runs in the late-bound _s16_prepare_delta chain, so the lab's arms rebuild it from their own source.
"""
import ast,hashlib,importlib.util,json,re,subprocess,sys,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path.insert(0,str(H.parent/'s16'));import patches
def _load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s18c=_load(H.parent/'s18/compose.py','s18_compose')
sha=lambda s:hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
S18_SHA='95df42fe7b3a31e66d60bc769bae15412c952ad77753b7d15b3beea3d89b3af4'
DEFAULT=['P10','RAW2','R8','D','FAST','LATE']
P10_ROWS={(8,2304):4}
WIN=R/'problems/muse_attention_decode/dev/window_rebuild';RING=R/'problems/muse_attention_decode/dev/full_attention/page_ring'
NEXT=H.parent/'nt1_timeline/review/next_issue';STRUCT=H.parent/'nt1_timeline/structural'
PIN={WIN/'P4_raw32.patch':None,WIN/'P4_pipe32.patch':None,RING/'R8_S18.cu':'5de7a696264fff0a',RING/'D_S18.cu':'36b40412cb5a4d79',
     RING/'S18_ATTN_PORT_CUDA_SRC.cu':'aeeb4aa8c00bfff0',RING/'S18_VQ8_CUDA.cu':'3a642563d3108946',
     NEXT/'base0_cell.py':'d82d404108f6a655',NEXT/'index0_cell.py':'75d6d0caac4605c7',NEXT/'fast0_cell.py':None,STRUCT/'split1_cell.py':None,
     R/'problems/muse_bf16_proj/dev/stream_gap/patch_qkv.py':None,R/'problems/muse_bf16_proj/dev/stream_gap/qkv_dispatch.py.inc':None}


def pins():
    out={}
    for p,want in PIN.items():
        h=sha(p.read_bytes());assert want is None or h.startswith(want),(p,h);out[str(p.relative_to(R))]=h
    return out


def hunk_lines(patch,i):
    """'+' lines (without '+') and context of the i-th hunk of a unified diff."""
    hunks=re.split(r'^@@.*@@\n',patch.read_text(),flags=re.M)[1:];return hunks[i]


def wr_block(patch):
    lines=[l[1:] for l in hunk_lines(patch,2).splitlines() if l.startswith('+')]
    text='\n'.join(lines).strip('\n')+'\n';assert text.startswith('wr_KERNELS=') and '_WINDOW_SELECTION=' in text;return text


def fast_index_cell(prefix='rvc_nfidx0'):
    """FAST's W2 cancellation (finalizer) and INDEX's B-fill touch disjoint CUDA lines of the SHARD2 cell."""
    norm=lambda p,x:(NEXT/p).read_text().replace(x,'xpfx').replace(x.upper(),'XPFX')
    b,i,f=norm('base0_cell.py','rvc_nbase0'),norm('index0_cell.py','rvc_nindex0'),norm('fast0_cell.py','rvc_nfast0')
    def cuda(t):
        (n,)=[n for n in ast.parse(t).body if isinstance(n,ast.Assign) and n.targets[0].id=='XPFX_CUDA'];return n,ast.literal_eval(n.value)
    with tempfile.TemporaryDirectory(dir=H/'generated') as d:
        paths=[]
        for name,t in (('i',i),('b',b),('f',f)):
            p=Path(d)/name;p.write_text(cuda(t)[1]);paths.append(str(p))
        z=subprocess.run(['git','merge-file','-p',*paths],capture_output=True,text=True);assert z.returncode==0,'FAST/INDEX CUDA conflict'
    node,_=cuda(f);merged=f.replace(ast.get_source_segment(f,node),'XPFX_CUDA = '+repr(z.stdout),1)
    return merged.replace('xpfx',prefix).replace('XPFX',prefix.upper())


def pdl_merge(code,prefix,cells,fast=False):
    change=next(x for x in cells['pdl_M'] if x['name']=='SMALLM_VAR_CUDA')
    a=ast.literal_eval(ast.parse(change['old']).body[0].value);b=ast.literal_eval(ast.parse(change['new']).body[0].value)
    if fast:
        # FAST moves gs2 into its per-expert float4 cancellation metadata and deletes the gs2 scale line;
        # pdl_M keeps only its gs13 preload there (its gs2 preload would read the float4 array).
        pre='  const float pa_gs13 = pa_f32(gs13+lane), pa_gs2 = pa_f32(gs2+lane);\n';sc='      const float sc = __shfl_sync(0xffffffffu,pa_gs2,e) * 256.f;\n'
        assert b.count(pre)==1 and b.count(sc)==1
        b=b.replace(pre,'  const float pa_gs13 = pa_f32(gs13+lane);\n').replace(sc,'      const float sc = gs2[e] * 256.f;\n')
    names=patches.bindings(code);cuda_name=prefix.upper()+'_VAR_CUDA' if prefix.upper()+'_VAR_CUDA' in names else prefix.upper()+'_CUDA'
    v=ast.literal_eval(names[cuda_name][0].value);return s18c.assign(code,cuda_name,patches.merge(a,b,v))


def s18_cells():
    raw=(H.parent/'s18/cells.json').read_bytes();return json.loads(raw)


def cells_pdl_M():
    return s18_cells()['pdl_M']


def delta(source,name,body):
    """Append one step to the late-bound prepare delta, timed into state['s19_prepare_s']."""
    return source+f'''
_s19_prior_{name}=_s16_prepare_delta
def _s16_prepare_delta(weights,state):
    state=_s19_prior_{name}(weights,state)
    import time as _t
    _t0=_t.monotonic()
{body}
    state.setdefault('s19_prepare_s',{{}})['{name}']=_t.monotonic()-_t0
    return state
'''


def redef(source,name,text):
    """Replace the one top-level def of name (with its decorators) by text ('' deletes it)."""
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name];assert len(nodes)==1,name
    n=nodes[0];lines=source.splitlines(True);start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    return ''.join(lines[:start])+text+''.join(lines[n.end_lineno:])


def swap(source,old,new):
    assert source.count(old)==1,old[:120];return source.replace(old,new,1)


# Oracle, tuner and fixture helpers that the bake leaves unreferenced; drop_dead deletes those nothing names.
DEAD=['_tune_glue','_glue_oracle','_best7_gate_moe','_best7_gate_attention','_s4_gate_moe','_s6_gate_moe','_measure_graph',
      '_best5_graph','_best5_fp32','_best5_fixture','_best5_refresh','_best6_fixture','_best6_refresh','_best7_moe_fixture',
      '_attention_public_draw','_attn_port_candidates','_best6_build_adapters','_best5_rms','_projection_consumer']


def drop_dead(source,names=DEAD):
    while True:
        tree=ast.parse(source);used={n.id for n in ast.walk(tree) if isinstance(n,ast.Name)}
        dead=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names and n.name not in used]
        if not dead:return source
        source=redef(source,dead[0],'')


BAKED={
'_best5_tune_moe':'''def _best5_tune_moe(state,weights,flush,device):
    # H200 28b5c189 BEST5_NT (measured): NT1 at M32/M128, NT2 at M256. Every BEST7 row and _s15n_plan names its
    # expert (nt1_lop3 / hh_m128 / mixed_parallel_m256), so only the summaries read this.
    state['moe_nt_plan']={32:1,128:1,256:2}
    state['moe_nt_rows']={m:dict(selected=nt,reason='H200 28b5c189') for m,nt in state['moe_nt_plan'].items()}
''',
'_best6_tune':'''def _best6_tune(state,weights,flush,device):
    # Every MoE shape runs its BEST7/_s15n expert: no optional backend is built or timed.
    state['moe6_plan']={};state['moe6_rows']={}
''',
'_tune_projections':'''def _tune_projections(state, flush, device):
    """H200 28b5c189 picks (BEST6_BASE_SHAPE): PROJ_PINS plus the measured cuBLAS at (1,'o'); no timing.
    Short-Lt pins run 'custom' until _tune_measured_lt binds them; Lt picks need Hopper's cuBLASLt module."""
    plan={};hopper=_fa3_on_device(device);real=torch.cuda.get_device_capability(device)==(9,0)
    if hopper:state["lt_mod"]=_lt_module();state["lt_workspace"]=torch.empty((32<<20,),dtype=torch.uint8,device=device)
    for t in _TUNE_TOKENS:
        for name in ("qkv_gate","o","pool_router","unpooler_slice" if t<=8 else "unpooler"):
            pick=PROJ_PINS.get((t,name),"cublas" if (t,name)==(1,"o") else "custom")
            # Off Hopper (local screens) tuned picks fall back to the custom GEMV, as the tuner's rejects did; VDSK's
            # tile needs 172 KB of shared memory, so it also needs a real Hopper under the forced screen.
            if isinstance(pick,tuple) and (pick[0]=="short_lt" or not hopper or pick[0]=="vdsk" and not real):pick="custom"
            if isinstance(pick,tuple) and pick[0]=="lt":
                w=state[name];x=torch.zeros((t,w.shape[1]),device=device,dtype=torch.bfloat16)
                out=torch.empty((t,w.shape[0]),device=device,dtype=torch.bfloat16)
                state.setdefault("lt_algos",{})[(name,t)]=state["lt_mod"].lt_options(w,x,out)
            plan[(t,name)]=pick
    return plan
''',
'_tune_measured_lt':'''_MEASURED_LT_LIB=[]
def _measured_lt_build():
    """PG_RAW for the Short-Lt pins, compiled in the native pool."""
    if _MEASURED_LT_LIB:return
    import hashlib
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    tag=hashlib.sha256(PG_RAW.encode()).hexdigest()[:12]
    _MEASURED_LT_LIB.append(load_inline(name=f'muse_decode_measured_lt_{tag}_{arch}',cpp_sources='',cuda_sources=PG_RAW,
        functions=None,is_python_module=False,no_implicit_headers=True,
        extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'],extra_ldflags=['-lcublasLt'],verbose=False))


def _tune_measured_lt(state,flush,device):
    """Bind the Short-Lt pins (H200 28b5c189 MEASURED_LT picks); no timing."""
    if not _fa3_on_device(device):return
    _measured_lt_build()
    bank={'mod':_PgNative(_MEASURED_LT_LIB[0]),'layouts':{},'workspace':torch.empty(4<<30,device=device,dtype=torch.uint8)}
    for (t,name),pick in PROJ_PINS.items():
        if not (isinstance(pick,tuple) and pick[0]=='short_lt'):continue
        _,nn,ordinal=pick;w=state[name];n,k=w.shape
        codes,_=bank['mod'].options(t,n,k,nn,name=='pool_router',bank['workspace'].numel())
        if ordinal>=len(codes):raise RuntimeError(f'Short-Lt pin {pick} unavailable at t={t} {name}')
        bank['layouts'][(t,name,nn)]=(w.T.contiguous() if nn else w,codes);state['proj_plan'][(t,name)]=pick
    if bank['layouts']:state['measured_lt']=bank
''',
'_tune_attention_portfolio':'''def _tune_attention_portfolio(state,flush,device):
    # Every BEST7 row names its attention and _s15n_plan maps each shape onto a BEST7 row: 'auto' is never read.
    state['attention_public_rows']=[]
    return {}
''',
'_s6_o_projection_op':'''def _s6_o_projection_op(state,device):
    op=Best7ProjectionCell(state['o'],1,6,fp32=False)
    x=torch.zeros((1,state['o'].shape[1]),device=device,dtype=torch.bfloat16)
    for _ in range(2):op(x)
    return op,dict(config=6,splits=1,source='exact95ab383d M1 O unsplit projection')
''',
'_vq8_prepare':'def _vq8_prepare(device):\n    return _VQ8_ROWS\n',
'_cluster_prepare':'def _cluster_prepare(device):\n    return _CLUSTER_ROWS\n',
'_partial_port_prepare':'def _partial_port_prepare(device):\n    return _PP_ROWS\n',
'_r2_prepare':'''def _r2_build():
    for kind in {arm[0] for arm in _R2_SELECTION.values()}:r2_module(kind)


def _r2_prepare(device):
    return _R2_SELECTION
''',
}


def bake(source):
    """Prepare keeps the compiles (one parallel pool, plus NF's coding thread) and the weight coding. The H200-measured
    picks (S18's scored 28b5c189 log) are constants; the private FP32 oracles and gates are gone (TASK's checker is the
    only correctness gate); compiles nothing reads (XQA/P8 attention, prefill adapters) are no longer built."""
    for name,text in BAKED.items():source=redef(source,name,text)
    source=swap(source,"        grouped_future=native_pool.submit(_pf_grouped_module)\n",'')
    source=swap(source,"        try:grouped_future.result()\n        except Exception as exc:_BEST6_GROUPED_ERROR=_best6_reason(exc)\n",'')
    source=swap(source,"""            attention_futures=[native_pool.submit(_attn_port_ext),
                native_pool.submit(_attn_port_xext,'group'),native_pool.submit(_attn_port_xext,'warp'),
                native_pool.submit(_attn_port_p8ext)]
""","            attention_futures=[native_pool.submit(_attn_port_ext)]\n")
    source=swap(source,"""        try:state['moe6_adapters']=_best6_build_adapters(weights)
        except Exception as exc:state['moe6_adapters']={'available':[],'rejected':{name:_best6_reason(exc) for name in ('raw17','humming_masked','dg_w8')}}
""","        state['moe6_adapters']={'available':[],'rejected':{}}\n")
    source=swap(source,"    _tune_glue(state,flush,device)\n",'')
    source=swap(source,"""            if moe!='auto':
                gate_key=('moe',tokens,moe)
                try:
                    if gate_key not in gates:gates[gate_key]=_best7_gate_moe(moe,tokens,length,state,weights)
                except Exception as exc:
                    if moe.startswith('v9nt') or moe in ('mixed_parallel_m256','nt1_lop3','hh_m128'):raise
                    plan['moe']='auto'
                    print(f'BEST7_REJECT key={key} expert={moe} reason={_best6_reason(exc)}',flush=True)
            if attention!='auto':
                gate_key=('attention',length,window,str(attention),glue)
                if gate_key not in gates:gates[gate_key]=_best7_gate_attention(attention,length,window,state,plan,device)
""",'')
    # NF port coding needs only the weights: it runs beside the base prepare, which no longer captures graphs.
    source=swap(source,"""def prepare(weights):
    state=_nf_port_base_prepare(weights)
    state['nf_port']=nf_prepare(weights,None if NF_PORT_FORCE else NF_PORT_SELECTED)
    return state
""","""def prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        nf=pool.submit(nf_prepare,weights,None if NF_PORT_FORCE else NF_PORT_SELECTED)
        state=_nf_port_base_prepare(weights)
        state['nf_port']=nf.result()
    return state
""")
    source+='\n_s16_native_builds.extend([_measured_lt_build,_r2_build])\n'
    return drop_dead(source)


WARM='''    _s19_warm(state,weights['qkv_gate'].device)'''
WARM_FN='''
def _s19_warm(state,device):
    """Run every timed shape once at ctx 257: every module loads and every kernel JITs in prepare, never in a capture."""
    global _ap_fa3_scheduler_metadata
    saved=_ap_fa3_scheduler_metadata
    if not _fa3_on_device(device):_ap_fa3_scheduler_metadata=lambda *_,**__:None   # FA3 is Hopper-only, as in TASK's local screen
    try:
        for (b,l,w) in [k for k in BEST7_SETTINGS if k!='other']:
            kernel_fn(_ap_generate_input(batch=b,ctx=257,layer='sliding' if w else 'full',seed=1,q_len=l,
                spread='agentic' if b>1 else 'fixed',dist='soft',page=1,device=device),state)
    finally:_ap_fa3_scheduler_metadata=saved
    torch.cuda.synchronize(device)
'''


def build(enabled=None,base18=None,window='RAW2',split2=False,p10=P10_ROWS):
    enabled=set(DEFAULT if enabled is None else enabled);assert enabled<=set(DEFAULT)
    source,meta18=s18c.build(base18);pdl='pdl_M' in meta18['enabled']
    if base18 is None:assert meta18['sha256']==S18_SHA
    cells=s18_cells()
    if 'FAST' in enabled:
        assert 'INDEX1' in meta18['enabled']
        old=cells['small_index'];old=pdl_merge(old,'rvc_nindex0',cells) if pdl else old
        new=fast_index_cell();new=pdl_merge(new,'rvc_nfidx0',cells,fast=True) if pdl else new
        assert source.count('\n'+old)==1;source=source.replace('\n'+old,'\n'+new,1)
        source=source.replace('rvc_nindex0_','rvc_nfidx0_');assert 'rvc_nindex0' not in source and 'RVC_NINDEX0' not in source
    if 'D' in enabled:
        # D_S18.cu is S18's post-exec VQ8 source (Vwait16/32 and pdl_M) plus the denominator WGMMA.
        assert pdl,'D is ported onto the pdl_M VQ8 source'
        current=ast.literal_eval(patches.bindings(source)['VQ8_CUDA'][0].value);assert sha(current).startswith(PIN[RING/'S18_VQ8_CUDA.cu'])
        value=(RING/'D_S18.cu').read_text();source=s18c.assign(source,'VQ8_CUDA',value)
        node=patches.bindings(source)['_vq8_module'][0];old=patches.segment(source,node)
        assert old.count('muse_s18_vq_'+sha(current)[:12])==1;source=source.replace(old,old.replace('muse_s18_vq_'+sha(current)[:12],'muse_s19_vq_'+sha(value)[:12]),1)
    if 'R8' in enabled:
        # In S18's dispatch the port kernel's attn_forward runs only at decode 16/32 full (cluster, partial-port,
        # r2 and VQ8 rows take every other shape), so R8 replaces its source: no second compile in prepare.
        current=ast.literal_eval(patches.bindings(source)['ATTN_PORT_CUDA_SRC'][0].value);assert sha(current).startswith(PIN[RING/'S18_ATTN_PORT_CUDA_SRC.cu'])
        source=s18c.assign(source,'ATTN_PORT_CUDA_SRC',(RING/'R8_S18.cu').read_text())
        assert "name=f'muse_decode_attn_port_{hashlib.sha256(ATTN_PORT_CUDA_SRC.encode()).hexdigest()[:12]}_{arch}'" in patches.segment(source,patches.bindings(source)['_attn_port_ext'][0])
    # S19's new compiles run beside the base prepare instead of in its pool, whose results the tuners wait for
    # (S18 prepare 241.6 s against the tuners' 235 s deadline); the first S19 delta step joins them.
    source+='''
_S19_BUILDS=[]
_s19_build_futures=[]
_s19_origin=_s16_origin_prepare
def _s16_origin_prepare(weights):
    from concurrent.futures import ThreadPoolExecutor
    if _S19_BUILDS:
        pool=ThreadPoolExecutor(max_workers=len(_S19_BUILDS));_s19_build_futures[:]=[pool.submit(fn) for fn in _S19_BUILDS];pool.shutdown(wait=False)
    return _s19_origin(weights)
'''
    source=delta(source,'join','    for future in _s19_build_futures:future.result()')
    if 'RAW2' in enabled:
        head="def _best7_attention(qkv,data,state,plan):\n    cfg=data['config'];choice=plan['attention']\n";assert source.count(head)==1
        source=source.replace(head,head+"""    if 0<cfg['window']<=2048 and (cfg['batch'],cfg['q_len']) in state.get('wr_rows',{}):
        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
        name,early=state['wr_rows'][(cfg['batch'],cfg['q_len'])]
        return wr_attention(dict(data,config=cfg),qkv,name,early).view(-1,Q_SIZE)
""",1)
        block=wr_block(WIN/('P4_raw32.patch' if window=='RAW2' else 'P4_pipe32.patch'))
        source+='\n'+block+'\ndef _s19_wr_build():\n    for name in wr_KERNELS:wr_module(name)\n_S19_BUILDS.append(_s19_wr_build)\n'
        source=delta(source,'RAW2',"    state['wr_rows']=_window_prepare(weights['qkv_gate'].device)")
    if split2:
        # P3a SPLIT2 alternative at M=32 only: separate W13/W2/combine launches (structural split1 cell).
        code=(STRUCT/'split1_cell.py').read_text();code=pdl_merge(code,'rvc_struct_split1',cells) if pdl else code
        source+='\n'+code+'\n_s16_native_builds.append(rvc_struct_split1_module)\n_s19_smallm=smallm_experts\n'
        source+='def smallm_experts(data,state,flags,stages=2,grid=264):\n    if data["hidden_states"].shape[0]==32:return rvc_struct_split1_experts(data,state,flags,2,grid)\n    return _s19_smallm(data,state,flags,stages,grid)\n'
        source+='_s19_split_banks=_s4_prepare_banks\ndef _s4_prepare_banks(state,weights):\n    _s19_split_banks(state,weights)\n    rvc_struct_split1_bind(state["s4_nt1_bank"],{8:69})\n'
        source=delta(source,'SPLIT2','    rvc_struct_split1_bind(state["s4_nt1_bank"],{8:69})')
    if 'P10' in enabled:
        qkv=_load(R/'problems/muse_bf16_proj/dev/stream_gap/patch_qkv.py','p10_patch_qkv')
        # The lab's arms rebuild state through the delta chain, so P10's bank coding moves from a prepare wrapper into it.
        wrapper=["_sgq_layer_prepare=prepare\n","def prepare(weights):\n    st=_sgq_layer_prepare(weights)\n    st['sgq']=sgq_prepare(weights,SGQ_LOCAL_REDUCE,SGQ_SELECTED)\n    return st\n\n"]
        out=qkv.patch(source)
        for w in wrapper:assert out.count(w)==1,w;out=out.replace(w,'')
        source=swap(out,'SGQ_SELECTED={(8,2304):4,(16,2304):4,(16,1280):4}','SGQ_SELECTED='+repr(p10))
        source+='''
def _s19_sgq_build():
    ns,insts,plans=sgq_source(SGQ_LOCAL_REDUCE,SGQ_SELECTED);ns['v18s_build'](insts,'sgq_'+hashlib.sha256(ns['v18s_cuda_source'](insts).encode()).hexdigest()[:12])
_S19_BUILDS.append(_s19_sgq_build)
'''
        source=delta(source,'P10',"    state['sgq']=sgq_prepare(weights,SGQ_LOCAL_REDUCE,SGQ_SELECTED)")
    if 'LATE' in enabled:
        # pdl_M hoisted four 8192-wide weight vectors above _boundary_late's PDL wait (T=256 glue only).
        assert pdl;late=next(x for x in cells_pdl_M() if x['name']=='_boundary_late');source=swap(source,late['new'],late['old'])
    source=delta(bake(source)+WARM_FN,'warm',WARM)
    compile(source,'<S19>','exec')
    meta=dict(base_sha256=meta18['sha256'],base18=meta18['enabled'],sha256=sha(source),enabled=sorted(enabled),window=window,split2=split2,
              p10=sorted(p10.items()) if 'P10' in enabled else None,baked='28b5c189')
    return source,meta


if __name__=='__main__':
    s,m=build();(H/'generated/s19.py').write_text(s);m['pins']=pins();(H/'selection.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=1)[:800])
