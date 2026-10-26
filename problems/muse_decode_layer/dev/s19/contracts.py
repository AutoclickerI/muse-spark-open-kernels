"""Static contract preflight for S19 and its lab arms (no GPU): run before any screen or send.

In the lab every arm (INC = frozen S18 included) starts from a copy of the host's base prepare state and rebuilds only
its own _s16_prepare_delta chain. S18's three integration failures were implicit prepare state (pool128 rebinding
prepare to the old chain, a flags table without M=8, the incumbent inheriting S18's attention plans), so:

1 flags   every flags word the small-M dispatch can pass to a cell is instantiated in that cell and bound on the bank
          the dispatch reads (s4_nt1_bank), for every token count the nt1_lop3 expert serves.
2 chain   no component rebinds prepare or the origin; each S19 delta step calls its predecessor; S19's prepare defs are
          S18's except the bake's NF wrapper.
3 plans   every plan table the base prepare reads is the same text in host and arm (delta-owned tables excepted), and the
          host's baked picks reproduce S18's own scored prepare log (28b5c189).
4 banks   every function and table that builds the shared base state is the same text in host and arm; each key the
          binds add to the shared bank has one producer.
Each S19 component's needs and products are declared in CONTRACTS; the checks read the source, not the declarations.
"""
import ast,json,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(H.parent/'s16')]
import compose
RECEIPT=R/'runs/muse_decode_layer/20261026T050310Z_submit_28b5c189e7b5469c.json'
CONTRACTS={
 'P10':dict(needs=['weights.qkv_gate'],produces=['sgq'],owner='delta P10',tables=['SGQ_SELECTED']),
 'RAW2':dict(needs=['plan rows (32,8) sliding'],produces=['wr_rows'],owner='delta RAW2',tables=['_WINDOW_SELECTION']),
 'R8':dict(needs=["attention ('attn_main',None) at d16F/d32F"],produces=[],owner='source ATTN_PORT_CUDA_SRC'),
 'D':dict(needs=['vq8_rows'],produces=[],owner='source VQ8_CUDA'),
 'FAST':dict(needs=['s4_nt1_bank.gs2','s4_nt1_bank.gs13','SMALLM_FLAGS'],produces=['s4_nt1_bank.cancel_meta','s4_nt1_bank.rvc_nfidx0_counters'],owner='delta bind'),
 'LATE':dict(needs=[],produces=[],owner='source _boundary_late'),
 'bake':dict(needs=['PROJ_PINS','BEST7_SETTINGS','CD_ROWS'],produces=['proj_plan','measured_lt','moe_nt_plan','moe6_plan','attention_plan',
             'vq8_rows','cluster_rows','r2_rows','partial_port_rows','nf_port'],owner='base prepare'),
}
DELTA_TABLES={'SGQ_SELECTED','_WINDOW_SELECTION'}
BANK_ROOTS=['_prepare_weights','_prepare_recoded_bank','_s4_prepare_banks','_cd_prepare','nf_prepare','_gate_split_prepare','_seed_v9_gemv_choices']
TABLE=re.compile(r'^_?[A-Z][A-Z0-9_]*$')
TOKENS=sorted({1,8,16,32}|{b*q for b,q in ((1,1),(8,1),(9,1),(3,8),(5,8),(33,1),(2,8),(4,8),(16,1),(1,8)) if b*q<=64})


class Src:
    def __init__(self,text):
        self.text=text;self.tree=ast.parse(text);self.lines=text.splitlines(True)
        self.defs={};self.tables={};self.env={}
        for n in self.tree.body:
            if isinstance(n,ast.FunctionDef):self.defs.setdefault(n.name,[]).append(n)
            targets=[t.id for t in (n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,ast.AugAssign) else []) if isinstance(t,ast.Name)]
            for t in targets:
                if TABLE.match(t) and not (isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)):self.tables.setdefault(t,[]).append(self.seg(n))
                if isinstance(n,ast.Assign) and isinstance(n.value,ast.Name):self.env[t]=n.value.id
    def seg(self,n):
        return ''.join(self.lines[min([n.lineno]+[d.lineno for d in getattr(n,'decorator_list',[])])-1:n.end_lineno])
    def segs(self,name):return [self.seg(n) for n in self.defs.get(name,[])]
    def literal(self,name):
        node=[n for n in self.tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets)][-1]
        return ast.literal_eval(node.value)


def check_flags(s):
    out=[];smallm=s.literal('SMALLM_FLAGS')
    call="SMALLM_FLAGS[1 if data['hidden_states'].shape[0]==1 else 64]"
    assert any(call in seg for seg in s.segs('_best6_call')),'nt1_lop3 flags rule changed: update check_flags'
    # The cell each dispatch level sends M to; a level's last return falls through to an alias of the previous level.
    def route_node(node,m):
        for st in node.body:
            if isinstance(st,ast.If) and isinstance(st.test,ast.Compare) and isinstance(st.test.ops[0],ast.In):
                if m in ast.literal_eval(st.test.comparators[0]):return st.body[0].value.func.id
            if isinstance(st,ast.Return) and isinstance(st.value,ast.Call):
                name=st.value.func.id
                if s.env.get(name)!='smallm_experts':return name
                prior=[n for n in s.tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets)][-1]
                levels=[n for n in s.defs['smallm_experts'] if n.lineno<prior.lineno]
                return route_node(levels[-1],m) if levels else 'smallm_base'
        return 'smallm_base'
    bound={}
    for n in ast.walk(s.tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id.endswith('_bind') and len(n.args)==2:
            bank=ast.unparse(n.args[0]).replace('"',"'");arg=n.args[1]
            flags=s.literal(arg.id) if isinstance(arg,ast.Name) else ast.literal_eval(arg)
            bound.setdefault(n.func.id[:-5],[]).append((bank,set(flags.values())))
    for m in TOKENS:
        flags=smallm[1 if m==1 else 64];cell=route_node(s.defs['smallm_experts'][-1],m)
        prefix='smallm' if cell in ('smallm_base',) else cell[:-len('_experts')] if cell.endswith('_experts') else cell
        if prefix=='smallm':built=s.literal('SMALLM_FLAGS_BUILT');bank_ok=[b for b,f in bound.get('smallm',[]) if flags in f]
        else:
            bind=s.defs.get(prefix+'_bind',[None])[-1]
            if bind is None:
                out.append(dict(check='flags',M=m,cell=cell,ok=True,note='cell without per-flags state'));continue
            names=[c.comparators[0].id for c in ast.walk(bind) if isinstance(c,ast.Compare) and isinstance(c.ops[0],ast.NotIn) and isinstance(c.comparators[0],ast.Name)]
            built=s.literal(names[0]) if names else None
            bank_ok=[b for b,f in bound.get(prefix,[]) if flags in f and b=="state['s4_nt1_bank']"]
        ok=built is not None and flags in built and bool(bank_ok)
        out.append(dict(check='flags',M=m,cell=cell,flags=flags,built=sorted(built) if built else None,bound=bool(bank_ok),ok=ok))
    for prefix,binds in bound.items():
        bind=s.defs.get(prefix+'_bind',[None])[-1]
        names=[c.comparators[0].id for c in ast.walk(bind) if isinstance(c,ast.Compare) and isinstance(c.ops[0],ast.NotIn) and isinstance(c.comparators[0],ast.Name)] if bind else []
        if names:
            built=set(s.literal(names[0]))
            for bank,flags in binds:out.append(dict(check='bind_built',cell=prefix,flags=sorted(flags),ok=flags<=built))
    return out


def check_chain(s,s18):
    out=[];expect=s18.segs('prepare')
    for old,new in BAKE_SWAPS():expect=[x.replace(old,new) for x in expect]
    out.append(dict(check='prepare_defs',ok=s.segs('prepare')==expect,n=len(s.segs('prepare'))))
    rebinds=[n.lineno for n in s.tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='prepare' for t in n.targets)]
    out.append(dict(check='prepare_assign',ok=not rebinds,lines=rebinds))
    origins=s.segs('_s16_origin_prepare')[len(s18.segs('_s16_origin_prepare')):]
    out.append(dict(check='origin_wrapper',ok=len(origins)==1 and 'return _s19_origin(weights)' in origins[0]))
    steps=[n for n in s.tree.body if isinstance(n,ast.FunctionDef) and n.name=='_s16_prepare_delta'][len(s18.defs['_s16_prepare_delta']):]
    for n in steps:
        first=ast.unparse(n.body[0]);m=re.fullmatch(r'state = (_s19_prior_\w+)\(weights, state\)',first)
        alias=[a for a in s.tree.body if isinstance(a,ast.Assign) and m and any(getattr(t,'id','')==m.group(1) for t in a.targets)]
        ok=bool(m) and len(alias)==1 and alias[0].value.id=='_s16_prepare_delta' and alias[0].lineno<n.lineno
        out.append(dict(check='delta_step',step=m.group(1)[11:] if m else first,ok=ok))
    return out


def BAKE_SWAPS():
    import inspect
    src=inspect.getsource(compose.bake);tree=ast.parse(src)
    return [(ast.literal_eval(c.args[1]),ast.literal_eval(c.args[2])) for c in ast.walk(tree)
            if isinstance(c,ast.Call) and getattr(c.func,'id','')=='swap']


def receipt_plans():
    log=json.loads(RECEIPT.read_text())['result']['log_tail'];base={};plans={}
    for line in log.splitlines():
        if line.startswith('BEST6_BASE_SHAPE'):
            b,q=int(re.search(r'b=(\d+)',line).group(1)),int(re.search(r'q=(\d+)',line).group(1))
            base[b*q]=ast.literal_eval(re.search(r'proj=(\{.*?\}) prefix',line).group(1))
        if line.startswith('BEST7_PLAN'):
            p=json.loads(line.split(' ',1)[1]);plans[tuple(p['key'])]=p
    return base,plans


def baked_proj(s):
    pins=s.literal('PROJ_PINS');cd=s.literal('CD_ROWS');plan={}
    for t in s.literal('_TUNE_TOKENS'):
        for name in ('qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
            pick=pins.get((t,name),'cublas' if (t,name)==(1,'o') else 'custom')
            plan[t,name]='custom' if (t,name) in cd else pick   # Hopper: Lt and Short-Lt pins bind as pinned
    return plan


def jsonable(v):
    return json.loads(json.dumps(v))


def check_plans(s,arm,name):
    out=[]
    for table in sorted(set(s.tables)|set(arm.tables)):
        if table in DELTA_TABLES or table not in s.tables or table not in arm.tables:continue
        out.append(dict(check='plan_table',arm=name,table=table,ok=s.tables[table]==arm.tables[table]))
    return out


def check_receipt(s):
    base,plans=receipt_plans();proj=baked_proj(s);out=[]
    for t,roles in base.items():
        for role,pick in roles.items():
            out.append(dict(check='receipt_proj',t=t,role=role,ok=jsonable(proj[t,role])==jsonable(pick),baked=str(proj[t,role]),receipt=str(pick)))
    settings=s.literal('BEST7_SETTINGS')
    for key,p in plans.items():
        glue,attention,moe,projection=settings[key];want=dict(proj)
        resolved={r:(projection.get(r,'auto') if isinstance(projection,dict) else 'auto') for r in p['projection']}
        resolved={r:want[key[0]*key[1],r] if v=='auto' else v for r,v in resolved.items()}
        out.append(dict(check='receipt_plan',key=key,ok=jsonable(resolved)==p['projection'] and jsonable(attention)==p['attention'] and moe==p['moe']))
    nt={int(m):int(sel) for m,sel in re.findall(r"BEST5_NT M=(\d+) \{.*?'selected': (\d)",json.loads(RECEIPT.read_text())['result']['log_tail'])}
    out.append(dict(check='receipt_nt',ok=nt=={32:1,128:1,256:2},receipt=nt))
    return out


def reach(s,roots):
    seen=set();todo=list(roots)
    while todo:
        name=todo.pop()
        if name in seen or (name not in s.defs and name not in s.tables):continue
        seen.add(name)
        for n in s.defs.get(name,[]):todo+=[x.id for x in ast.walk(n) if isinstance(x,ast.Name) and not x.id.endswith(('_bind','_module'))]
    return seen


def check_banks(s,arm,name):
    out=[]
    # Binds and the modules/flags they compile are per arm (each arm's delta rebinds its own cells on the copy).
    own=lambda fn:fn.endswith(('_bind','_module','_MODULE','_FLAGS_BUILT'))
    norm=lambda segs:[re.sub(r'(?m)^\s*\w+_bind\(.*\n','',x) for x in segs or []]
    for fn in sorted(n for n in reach(s,BANK_ROOTS) if not own(n)):
        a,b=(norm(s.segs(fn)) or s.tables.get(fn)),(norm(arm.segs(fn)) or arm.tables.get(fn))
        if a!=b:out.append(dict(check='bank_producer',arm=name,fn=fn,ok=False))
    writes={}
    for fn,nodes in arm.defs.items():
        if not fn.endswith('_bind'):continue
        for n in ast.walk(nodes[-1]):
            if isinstance(n,ast.Subscript) and isinstance(n.ctx,ast.Store) and ast.unparse(n.value)=='state':
                writes.setdefault(ast.unparse(n.slice),set()).add(fn)
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='setdefault' and ast.unparse(n.func.value)=='state':
                writes.setdefault(ast.unparse(n.args[0]),set()).add(fn)
    called={c.func.id for c in ast.walk(arm.tree) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id.endswith('_bind')}
    for key,fns in writes.items():
        live=fns&called
        out.append(dict(check='bank_key',arm=name,key=key,writers=sorted(live),ok=len(live)<=1))
    out.append(dict(check='bank_producers',arm=name,ok=not any(not c['ok'] for c in out if c['check']=='bank_producer'),n=len(reach(s,BANK_ROOTS))))
    return out


def check_components(s):
    out=[];sgq=s.literal('SGQ_SELECTED');configs=s.literal('SGQ_CONFIGS')
    out.append(dict(check='P10_rows',ok=all(t in (8,16) and n in (1280,2304) and 0<=a<len(configs) for (t,n),a in sgq.items()),rows=sorted(sgq)))
    wr=s.literal('_WINDOW_SELECTION');kernels=s.literal('wr_KERNELS')
    out.append(dict(check='RAW2_rows',ok=all(v[0] in kernels for v in wr.values()),rows=sorted(wr)))
    vq=s.segs('_vq8_module')[-1];vcuda=s.literal('VQ8_CUDA')
    out.append(dict(check='D_module_name',ok=('muse_s19_vq_'+compose.sha(vcuda)[:12]) in vq))
    late=next(x for x in compose.cells_pdl_M() if x['name']=='_boundary_late')
    out.append(dict(check='LATE_source',ok=s.segs('_boundary_late')==[late['old']]))
    return out


def run(units):
    s18=Src((H.parent/'s18/generated/s18.py').read_text());host=Src(units['S19']);checks=[]
    for name,text in units.items():
        src=Src(text);checks+=[dict(c,unit=name) for c in check_flags(src)]
        if name!='INC':checks+=[dict(c,unit=name) for c in check_chain(src,s18)]
        if name!='S19':checks+=check_plans(host,src,name)+check_banks(host,src,name)
    checks+=[dict(c,unit='S19') for c in check_receipt(host)+check_components(host)]
    return dict(contracts=CONTRACTS,checks=checks,passed=all(c['ok'] for c in checks))


if __name__=='__main__':
    build_lab=compose._load(H/'build_lab.py','s19_build_lab')
    units={'S19':compose.build()[0],'INC':(H.parent/'s18/generated/s18.py').read_text()}
    for name,(kw,_) in build_lab.ARMS.items():
        if kw is not None:units[name]=compose.build(**kw)[0]
    result=run(units);(H/'contracts.json').write_text(json.dumps(result,indent=1,default=str)+'\n')
    bad=[c for c in result['checks'] if not c['ok']]
    print('PASS' if result['passed'] else 'FAIL',len(result['checks']),'contract checks over',', '.join(units));[print(' ',c) for c in bad[:40]]
    sys.exit(0 if result['passed'] else 1)
