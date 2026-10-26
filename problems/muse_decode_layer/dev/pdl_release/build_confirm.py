"""Isolate the native reset signals; retain per-seed times for held-out R."""
import ast
import base64
import difflib
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[1];R=P.parents[1]
out=H/'confirmation';out.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('release_build',H/'build.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
base=b.BASE.read_text();full=b.compose()[0]

def node_text(src,name):
    n=next(n for n in ast.parse(src).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
    return ''.join(src.splitlines(True)[n.lineno-1:n.end_lineno]),n


def source(groups,extras=()):
    s=b.compose(without=tuple(g for g in ('attention','moe','projections') if g not in groups))[0]
    # Reset grouped small-M and generic V9 release edits: only the independently
    # selected large-M producer family is part of this confirmation.
    for name in ('SMALLM_VAR_CUDA','V9_CUDA_SRC'):
        old,n=node_text(base,name);_,t=node_text(s,name);lines=s.splitlines(True);lines[t.lineno-1:t.end_lineno]=[old];s=''.join(lines)
    def one(a,z):
        nonlocal s
        assert s.count(a)==1,(a,s.count(a));s=s.replace(a,z)
    if 'o' in extras:
        one('NF_PORT_FORCE = False',"NF_PORT_SELECTED |= {(128, 'o')}\nNF_PORT_FORCE = False")
    if 'pool' in extras:
        pre='' if 'projections' in groups else '        tl.extra.cuda.gdc_launch_dependents()\n'
        one('NS: tl.constexpr, TMA: tl.constexpr):\n',
            'NS: tl.constexpr, TMA: tl.constexpr, CHAIN: tl.constexpr = False):\n    if CHAIN:\n'+pre+'        tl.extra.cuda.gdc_wait()\n')
        one('def _vdsk_project(x, w, variant):','def _vdsk_project(x, w, variant, chain=False):')
        old='num_warps=NWARPS, num_stages=NS'+(', launch_pdl=True' if 'projections' in groups else '')+')\n    return P, S'
        one(old,'num_warps=NWARPS, num_stages=NS, CHAIN=chain, launch_pdl='+('True' if 'projections' in groups else 'chain')+')\n    return P, S')
        # The common early-release trigger precedes the dependent wait.
        if 'projections' in groups:
            one('    if CHAIN:\n        tl.extra.cuda.gdc_wait()\n    tl.extra.cuda.gdc_launch_dependents()\n',
                '    tl.extra.cuda.gdc_launch_dependents()\n    if CHAIN:\n        tl.extra.cuda.gdc_wait()\n')
        marker='def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):\n    key=(x.shape[0],name)\n'
        one(marker,"LIBRARY_NATIVE_ROWS = {(128, 'pool_router'): ('vdsk',(128,128,64,5,8,1),3)}\n\n"+marker+
            '    if key in LIBRARY_NATIVE_ROWS:\n        return _vdsk_project(x,state[name],LIBRARY_NATIVE_ROWS[key],chain=True)\n')
    return s

arms=dict(P=('attention','moe','projections'),attention=('attention',),large=('moe',),vdsk=('projections',),
          no_attention=('moe','projections'),no_large=('attention','projections'),no_vdsk=('attention','moe'))
extras={name:() for name in arms}
for name,groups,lib in [('O',(),('o',)),('pool',(),('pool',)),('OP',(),('o','pool')),
                      ('C',arms['P'],('o','pool')),('C_no_o',arms['P'],('pool',)),('C_no_pool',arms['P'],('o',))]:
    arms[name]=groups;extras[name]=lib
specs={};meta={}
for name,groups in arms.items():
    s=source(groups,extras[name]);(out/(name+'.py')).write_text(s)
    diff=''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='cand_r3_best7_s15.py',tofile=name+'.py'))
    (out/(name+'.patch')).write_text(diff)
    edits=[(a,z,''.join(s.splitlines(True)[c:d])) for op,a,z,c,d in difflib.SequenceMatcher(None,base.splitlines(True),s.splitlines(True),autojunk=False).get_opcodes() if op!='equal']
    specs[name]=dict(edits=edits,sha256=hashlib.sha256(s.encode()).hexdigest())
    meta[name]=dict(groups=groups,library=extras[name],source_sha256=specs[name]['sha256'],patch_sha256=hashlib.sha256(diff.encode()).hexdigest())
raw=(P/'dev/library_chain/out/library_payload.py').read_text();n=next(n for n in ast.parse(raw).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_BUNDLE')
bundle=eval(compile(ast.Expression(n.value),'bundle','eval'),dict(base64=base64,lzma=lzma,json=json));bundle['arms']=specs
enc=base64.b85encode(lzma.compress(json.dumps(bundle,separators=(',',':')).encode())).decode()
body=(P/'dev/library_chain/lab.py.inc').read_text().replace('6107923+7919*i','9837141+7919*i')
a=body.index('    # Exactly one additional NF O preparation');z=body.index('    a=torch.cuda.Event(enable_timing=True)',a)
body=body[:a]+'''    # Modules are content-addressed; unchanged original preparation is shared.
    extra=spaces['O']['nf_prepare'](weights,{(128,'o')})
    nf=dict(state['nf_port'],plans=dict(state['nf_port']['plans']))
    nf['plans'].update(extra['plans'])
    from concurrent.futures import ThreadPoolExecutor
    def native(ns):
        ns['nf_module']()
        if ns['_attn_port_hopper'](weights['o'].device):ns['_attn_port_ext']()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={name:pool.submit(native,ns) for name,ns in spaces.items() if name not in ('A','A2')}
        for name,f in jobs.items():f.result();states[name]=dict(state,nf_port=nf)
'''+body[z:]
a=body.index('native_proof={name:dict(');z=body.index('    for batch,length,window in CASES:',a)
body=body[:a]+'''native_proof={name:dict(large=ns['NF_CUDA']!=base['NF_CUDA'],attention=ns['ATTN_PORT_CUDA_SRC']!=base['ATTN_PORT_CUDA_SRC'],
        smallm_unchanged=ns['SMALLM_VAR_CUDA']==base['SMALLM_VAR_CUDA'],v9_unchanged=ns['V9_CUDA_SRC']==base['V9_CUDA_SRC'])
        for name,ns in spaces.items() if name not in ('A','A2')}
'''+body[z:]
body=body.replace('elapsed=round(time.monotonic()-started,2),complete=',
    'seed_load_us={f\'{b}x{l}\':[{name:round((46*s[name]+16*f[name])/62,3) for name in s} '
    'for s,f in zip(samples[b,l,2048],samples[b,l,0])] for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)) '
    'if (b,l,2048) in samples and (b,l,0) in samples},elapsed=round(time.monotonic()-started,2),complete=')
body=body.replace('META r=arm/S15;L=all;no_*=L minus replacement;A2=A:A;8 seeds,8 cold graph replays',
    'META P=attention+largeMoE+VDSK release; OP=native O128/pool128; C=P+OP; single/leave-one-out arms; A:A;8 held-out seeds,8 cold replays, TASK checker, per-seed R inputs')
text='import torch,time,statistics,math,json,gc,ast,hashlib,linecache,types,base64,lzma\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(enc)+')))\n'+body
(out/'release_confirmation.py').write_text(text)
host=(P/'dev/s15/s15ab_host.py').read_text()+'\n_entry_kernel=globals()["kernel_fn"]\ndef kernel_fn(data,state):\n    return _entry_kernel(data,state)\n'
(out/'host.py').write_text(host)
(out/'manifest.json').write_text(json.dumps(dict(base_sha256=b.SHA,arms=meta,payload_sha256=hashlib.sha256(text.encode()).hexdigest(),
    task_checks=1440,decision='Nomination only until held-out paired six-load R lower confidence bound exceeds1; TASK checker is the sole correctness gate.'),indent=2)+'\n')
print(meta)
