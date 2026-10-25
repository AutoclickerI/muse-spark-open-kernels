"""Production patches for the reset's executable arms, rebased onto S15.

Only selected cells are emitted. Diagnostic ready-input/profiler controls
are intentionally not production candidates. Native code is pinned in cells.json.
"""
import ast
import difflib
import hashlib
import json
import re
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
BASE=R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
BASE_SHA='602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb'


def segment(source,node):
    return ''.join(source.splitlines(True)[min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1:node.end_lineno])


def rename(source,mapping):
    class Names(ast.NodeTransformer):
        def visit_Name(self,node):
            if node.id in mapping:node.id=mapping[node.id]
            return node
        def visit_FunctionDef(self,node):
            if node.name in mapping:node.name=mapping[node.name]
            return self.generic_visit(node)
        def visit_Global(self,node):
            node.names=[mapping.get(name,name) for name in node.names]
            return node
    tree=Names().visit(ast.parse(source));ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def bindings(source):
    result={}
    for node in ast.parse(source).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):names=[node.name]
        elif isinstance(node,ast.Assign):names=[t.id for target in node.targets for t in ast.walk(target) if isinstance(t,ast.Name)]
        else:continue
        for name in names:result.setdefault(name,[]).append(node)
    return result


def merge(old,new,target):
    """Asserted text merge for independent edits inside the same Python/CUDA node."""
    if target==old:return new
    a,b=old.splitlines(True),new.splitlines(True)
    for op,lo,hi,c,d in reversed(list(difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes())):
        if op=='equal':continue
        matched=False
        for context in (3,2,1,0):
            left=max(0,lo-context);right=min(len(a),hi+context)
            before=''.join(a[left:right])
            after=''.join(a[left:lo]+b[c:d]+a[hi:right])
            if before and target.count(before)==1:
                target=target.replace(before,after,1);matched=True;break
        assert matched,('overlapping patch',old[:60],''.join(a[lo:hi])[:140])
    return target


def apply_delta(source,changes):
    edits=[];added=[];nodes=bindings(source)
    for change in changes:
        old,new=change['old'],change['new']
        if old is None:added.append(new);continue
        name=change['name']
        if change['kind']=='Assign':
            parsed=ast.parse(old).body[0]
            key=next((t.id for t in parsed.targets if isinstance(t,ast.Name)),None)
        else:key=name
        candidates=nodes.get(key,[])
        exact=[n for n in candidates if ast.dump(n)==ast.dump(ast.parse(old).body[0])]
        node=exact[0] if len(exact)==1 else candidates[min(change['ordinal'],len(candidates)-1)] if candidates else None
        assert node is not None,(name,'missing target')
        current=segment(source,node)
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            before=ast.literal_eval(ast.parse(old).body[0].value)
            after=ast.literal_eval(ast.parse(new).body[0].value)
            updated=ast.unparse(node.targets[0])+' = '+repr(merge(before,after,node.value.value))+'\n'
        else:updated=merge(old,new,current)
        start=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])-1
        edits.append((start,node.end_lineno,updated))
    lines=source.splitlines(True)
    for lo,hi,new in sorted(edits,reverse=True):lines[lo:hi]=[new+'\n']
    return ''.join(lines)+'\n\n'+'\n\n'.join(added)+'\n'


def registry():
    arms={}
    def add(name,family,rows,**kwargs):arms[name]=dict(name=name,family=family,rows=rows,**kwargs)
    decode=[(1,1,True),(16,1,True),(32,1,True),(1,1,False)]
    for kind,nk,split in [('C64',64,None),('C128',128,None),('W4',128,4),('W8',128,8)]:
        add('global_'+kind,'global',decode,nk=nk,split=split)
    verify=[(b,8,True) for b in (1,16,32)]
    for gang in (1,2,4,8,16):add('dsm_'+str(gang),'dsm',verify,clusters={1:gang,16:gang,32:gang})
    add('dsm_wave','dsm',verify,clusters={1:16,16:4,32:2})
    for kind,qt,early in [('E4',4,True),('E2',2,True),('E1',1,True),('L4',4,False)]:
        add('early_'+kind,'early',[(1,1,True)],qt=qt,early=early)
    for kind in ('mix25','mix50','gap','mixgap','tc25','ping','ping25'):
        add('soft_'+kind,'soft',[(16,8,False),(32,8,False)],kind=kind)
    small=[(b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8)) for w in (True,False)]
    for kind in ('VEC','PROD','NORM','PAIR'):add('epi_'+kind,'epi',small,kind=kind)
    for kind in ('ONCE','ONCE_VEC','ONCE_PAIR'):add('reuse_'+kind,'reuse',small,kind=kind)
    for arm,kind in enumerate(('DIRECT_CLUSTER1','DIRECT_NORMAL','DIRECT_SERIAL','DEEP_SERIAL'),1):
        add('heavy_'+kind,'heavy',[(b,8,w) for b in (16,32) for w in (True,False)],arm=arm)
    for kind in ('R','P','RP','C','S'):add('boundary_'+kind,'boundary',[(b,1,w) for b in (16,32) for w in (True,False)] if kind in ('P','C','S') else [(b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8)) for w in (True,False)],kind=kind)
    for kind in ('B','M','D'):add('pdl_'+kind,'pdl',[(1,1,False)] if kind=='D' else [(b,q,w) for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) for w in (True,False)],kind=kind)
    return arms


def driver_cell(code,prefix):
    names=bindings(code)
    return rename(code,{name:prefix+name for name in names})


def patch(source,spec):
    cells=json.loads((H/'cells.json').read_text())['cells']
    family=spec['family'];name=spec['name'];prefix='_s16_'+re.sub(r'\W','_',name)+'_'
    rows=tuple(tuple(r) for r in spec['rows']);tail='';warm=[];setup=[]
    if '_s16_native_builds' not in bindings(source):
        anchor='    with ThreadPoolExecutor(max_workers=15) as native_pool:\n'
        assert source.count(anchor)==1
        source=source.replace(anchor,anchor+'        s16_futures=[native_pool.submit(fn) for fn in _s16_native_builds]\n',1)
        anchor='        glue_cl_future.result()\n';assert source.count(anchor)==1
        source=source.replace(anchor,anchor+'        for future in s16_futures:future.result()\n',1)
        tail+='\n_s16_native_builds=[]\n'
    if family in ('global','dsm','early'):
        if family=='global':
            cell=driver_cell(cells['global']['code'],prefix)
            fn=prefix+'cs_attention';module=prefix+'cs_module'
            cs="None" if spec['split'] is None else f"({spec['split']} if cfg['window'] else {spec['split']*8})"
            args=f"nk={spec['nk']},cs={cs}"
        elif family=='dsm':
            cell=driver_cell(cells['dsm']['code'],prefix);fn=prefix+'cc_attention';module=prefix+'cc_module'
            clusters={int(k):v for k,v in spec['clusters'].items()}
            args='nk=128,cs='+repr(clusters)+"[cfg['batch']]"
        else:
            cell=driver_cell(cells['early']['early' if spec['early'] else 'late'],prefix)
            fn=prefix+'attention';module=prefix+'module';args=f"nk=128,cs=16,qt={spec['qt']}"
        tail+=cell
        tail+=f'''
{prefix}base_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    cfg=data['config']
    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in {rows!r}:
        cfg=dict(cfg,num_q_heads=HQ,num_kv_heads=HKV,head_dim=HD)
        return {fn}(dict(data,config=cfg),qkv,{args}).view(-1,Q_SIZE)
    return {prefix}base_attention(qkv,data,state,plan)
'''
        warm.append(module+'()')
    elif family=='soft':
        cuda=cells['soft'][spec['kind']]
        tail+=prefix+'CUDA = '+repr(cuda)+'\n'+prefix+'MOD = None\n'
        tail+=f'''
def {prefix}module():
    global {prefix}MOD
    if {prefix}MOD is None:
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{{major}}{{minor}}'
        {prefix}MOD=load_inline(name='s16_soft_'+{hashlib.sha256(cuda.encode()).hexdigest()[:12]!r}+'_'+arch,cpp_sources=VQ8_CPP,cuda_sources={prefix}CUDA,
            functions=['vq8_forward','vq8_resources'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{{arch}},code=sm_{{arch}}'])
    return {prefix}MOD
'''
        node=bindings(source)['_vq8_attention'][-1]
        fn=segment(source,node).replace('def _vq8_attention(',f'def {prefix}attention(',1).replace('_vq8_module()',prefix+'module()')
        tail+=fn+'\n'
        tail+=f'''
{prefix}base_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
    cfg=data['config']
    if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in {rows!r}:
        return {prefix}attention(qkv,data,(6,8))
    return {prefix}base_attention(qkv,data,state,plan)
'''
        warm.append(prefix+'module()')
    elif family in ('epi','reuse'):
        cell=cells[family][spec['kind']];code=driver_cell(cell['code'],prefix);tail+=code
        fn=prefix+cell['prefix']+'_experts';module=prefix+cell['prefix']+'_module'
        setup.append(f"{prefix+cell['prefix']+'_bind'}(state['s4_nt1_bank']"+(',SMALLM_FLAGS)' if family=='epi' else ')'))
        tokens=sorted({b*q for b,q,w in rows})
        call=f'{fn}(data,state,flags,stages,grid)' if family=='epi' else f"{fn}(data,state,mode={cell['mode']})"
        tail+=f'''
{prefix}base_experts=smallm_experts
def smallm_experts(data,state,flags,stages=2,grid=264):
    if data['hidden_states'].shape[0] in {tokens!r}:return {call}
    return {prefix}base_experts(data,state,flags,stages,grid)
'''
        warm.append(module+'()')
    elif family=='heavy':
        part=cells['heavy'];arm=spec['arm'];cuda=part['cuda'][4 if arm==4 else 1]
        mapping={n:prefix+n for n in part['names']};mapping.update(nf_module=prefix+'module',m128_hh_pm_module=prefix+'module',NF_OPTS=prefix+'OPTS')
        tail+=prefix+'CUDA = '+repr(cuda)+'\n'+prefix+'CPP = '+repr(part['cpp'])+'\n'+prefix+'MOD=None\n'+prefix+'OPTS={}\n'
        tail+=f'''
def {prefix}module():
    global {prefix}MOD
    if {prefix}MOD is None:
        major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{{major}}{{minor}}'
        {prefix}MOD=load_inline(name='s16_hh_'+{hashlib.sha256(cuda.encode()).hexdigest()[:12]!r}+'_'+arch,cpp_sources={prefix}CPP,cuda_sources={prefix}CUDA,
            functions=['br_init','br_info','br_run','br_route','br_finish','pmz_init','pmz_smem','pmz_resident','pmz_run','oc_run','oc_info','oc_gather','oc_topology'],
            no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{{arch}},code=sm_{{arch}}'])
        {prefix}MOD.br_init();{prefix}OPTS['pmz']={prefix}MOD.pmz_init()
        for n in (64,128,192,256):
            for first in (0,1):{prefix}MOD.oc_info(n,first,1)
    return {prefix}MOD
'''
        tail+=rename(part['code'],mapping)
        tokens=sorted({b*q for b,q,w in rows})
        tail+=f'''
{prefix}base_call=nf_call
def nf_call(data,state):
    if data['hidden_states'].shape[0] not in {tokens!r}:return {prefix}base_call(data,state)
    plan={prefix}oc_plan(data,state['s5_hh_base'],state[{(prefix+'bank')!r}],state[{(prefix+'streams')!r}],{1 if arm==1 else 0},True,{arm>=3!r})
    for step in plan['steps']:out=step()
    return out
'''
        warm.append(prefix+'module()')
        setup += [f"state[{(prefix+'bank')!r}]={prefix}m128_hh_issue_nt_prepare(dict(weights,post_norm_w=weights['post_expert_norm']))",
                  f"state[{(prefix+'streams')!r}]=([torch.cuda.Stream() for _ in range(5)],torch.cuda.Event(),[torch.cuda.Event() for _ in range(5)])"]
    elif family in ('boundary','pdl'):
        delta=cells[family][spec['kind']]
        if family=='boundary' and spec['kind'] in ('C','S'):
            # The original lab's full-row prepare wrapper is replaced by this
            # patch's explicit selected-row setup below.
            delta=[d for d in delta if d['name'] not in ('prepare','_bd_base_prepare')]
        source=apply_delta(source,delta)
        # B/P are measured against S14's balanced unpooler. On S15 the NF port
        # owns those rows; do not claim an inactive hint as an implemented gain.
        if family=='boundary' and spec['kind'] in ('P','RP'):
            source=adapt_nf_hint(source)
            setup.append("updated=nf_prepare(weights,{key for key in NF_PORT_SELECTED if key[1]=='unpooler' and key[0] in (16,32)})")
            setup.append("state['nf_port']=dict(state['nf_port'],plans=dict(state['nf_port']['plans']))")
            setup.append("state['nf_port']['plans'].update(updated['plans'])")
        if family=='boundary' and spec['kind'] in ('C','S'):
            for key in rows:
                if spec['kind']=='C':setup += [f"state['best7_plans'][{key[0],key[1],2048 if key[2] else 0!r}]['geometry']['b1']=('cl',4,256,1)",f"state['best7_plans'][{key[0],key[1],2048 if key[2] else 0!r}]['geometry']['b2']=('cl',4,256,1)"]
                else:setup.append(f"state['best7_plans'][{key[0],key[1],2048 if key[2] else 0!r}]['bd_serial']=True")
        if 'GP_ROWS' in bindings(source):setup.append("\n".join("for key,plan in state['best7_plans'].items():\n    plan['gp_key']=key".splitlines()))
        if family=='boundary' and spec['kind']=='S':warm.append('bd_serial_module()')
        if family=='pdl' and spec['kind']=='D':warm.append('gp_module()')
    else:raise AssertionError(family)
    # Every patched source exposes one incremental setup hook, so labs can share
    # the base's immutable prepared banks without rerunning the original tuner.
    for call in warm:
        assert call.endswith('()')
        tail+=f'\n_s16_native_builds.append({call[:-2]})\n'
    if family in ('epi','reuse','heavy'):
        # These calls are also used by the incumbent's prepare-time FP32 expert
        # gates. Install new banks/counters before those gates, never fall through
        # to old math merely because final prepare has not returned yet.
        tail+=f'\n{prefix}base_banks=_s4_prepare_banks\ndef _s4_prepare_banks(state,weights):\n    {prefix}base_banks(state,weights)\n'
        for line in warm+setup:tail+='\n'.join('    '+s for s in line.splitlines())+'\n'
    if '_s16_origin_prepare' not in bindings(source):tail+='\n_s16_origin_prepare=prepare\n'
    tail+=f'\n{prefix}base_delta=globals().get("_s16_prepare_delta")\n'
    tail+=f'def _s16_prepare_delta(weights,state):\n    if {prefix}base_delta is not None:state={prefix}base_delta(weights,state)\n'
    for line in warm+setup:tail+='\n'.join('    '+s for s in line.splitlines())+'\n'
    tail+='    return state\n'
    tail+='\ndef prepare(weights):\n    return _s16_prepare_delta(weights,_s16_origin_prepare(weights))\n'
    out=source+'\n\n'+tail
    ast.parse(out)
    return out


def adapt_nf_hint(source):
    # Same immutable first-weight-line hint at the port's actual producer entry.
    # This rebase needs whole-layer pairing even if P7's S14 hint wins: the
    # selected NF producer owns these calls in S15 and has different scheduling.
    source+='''
_s16_hint_original_port_cuda=_port_cuda
def _port_cuda(src,**kwargs):
    out=_s16_hint_original_port_cuda(src,**kwargs)
    hint=''
    if kwargs.get('m32w'):
        hint=r"""
  {int s16_wg=threadIdx.x/128,s16_warp=(threadIdx.x%128)/32;
   int s16_k0=(k/64)*((blockIdx.x%split)*WK+s16_wg)/(split*WK);
   if(s16_warp<R/16){const uint4* s16_w=w+((long)((blockIdx.x/split)*(k/64)+s16_k0)*(R/16)+s16_warp)*(C12?3:4)*32+(threadIdx.x&31);
    asm volatile("prefetch.global.L2 [%0];"::"l"(s16_w):"memory");}}
"""
    elif 'dd_cap2_b_cpssm_gemv' in src:
        hint=r"""
  {int s16_wr=CWR?CWR:WR_arg,s16_wk=CWK?CWK:WK_arg;
   int s16_gx=CN?CN/(s16_wr*RT*16):gridDim.x;
   int s16_gw=(blockIdx.y*s16_gx+blockIdx.x)*(s16_wr*s16_wk)+threadIdx.x/32;
   const int4* s16_w=wp+(size_t)s16_gw*(RT*NL*32)+(threadIdx.x&31);
   asm volatile("prefetch.global.L2 [%0];"::"l"(s16_w):"memory");}
"""
    if hint:
        anchor='  asm volatile("griddepcontrol.wait;" ::: "memory");'
        assert out.count(anchor)==1
        out=out.replace(anchor,hint+anchor)
    return out
'''
    return source


def compose(names,base=None):
    source=BASE.read_text() if base is None else base
    assert hashlib.sha256(source.encode()).hexdigest()==BASE_SHA
    manifest=json.loads((H/'manifest.json').read_text())
    assert hashlib.sha256((H/'cells.json').read_bytes()).hexdigest()==manifest['cells_sha256']
    choices=registry();selected=[]
    for item in names:
        spec=choices[item] if isinstance(item,str) else item
        source=patch(source,spec);selected.append(spec)
    return source,dict(base_sha256=BASE_SHA,sha256=hashlib.sha256(source.encode()).hexdigest(),patches=selected)
