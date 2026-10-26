"""Single stage-host batch: all full loads, structural arms, TASK, clocks."""
import argparse,ast,base64,difflib,hashlib,json,lzma
from pathlib import Path
from lab_sources import host
ap=argparse.ArgumentParser();ap.add_argument('--page-ring',action='store_true');args=ap.parse_args()
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/('page_ring' if args.page_ring else 'roof')
base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();adapter=(H/'attention_adapter.py.inc').read_text()
def get(s,key):return ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key))
def clock(c):
    # Exact old/new contextual hunks apply also to the partial-input prologue.
    import re
    lines=(H/'q1_clock.patch').read_text().splitlines(True);i=2;hunks=[]
    while i<len(lines):
        assert lines[i].startswith('@@');i+=1;a=[];b=[]
        while i<len(lines) and not lines[i].startswith('@@'):
            x=lines[i];i+=1
            if x[0] in ' -':a.append(x[1:])
            if x[0] in ' +':b.append(x[1:])
        hunks.append((''.join(a),''.join(b)))
    for a,b in reversed(hunks):assert c.count(a)==1,(a[:60],c.count(a));c=c.replace(a,b)
    return c
arms={}
names=('G','R4','R8','D') if args.page_ring else ('H1','H2','F','G','P')
for name in names:
    path=H/'pvsum/candidate.py' if name=='D' else (H/'roof' if name=='G' else o)/(name+'.py')
    s=path.read_text();a,b=base.splitlines(True),s.splitlines(True)
    arms[name]=dict(edits=[(lo,hi,''.join(b[c:d])) for op,lo,hi,c,d in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes() if op!='equal'],sha256=hashlib.sha256(s.encode()).hexdigest())
p=R/'problems/muse_attention_decode'
reference='\n'.join(l for l in (p/'reference.py').read_text().splitlines() if not l.startswith('from checks import'))
checker='\n'.join(l for s in ((p/'checks.py').read_text(),reference) for l in s.splitlines() if not l.startswith('from __future__ import'))
checker+='\nCACHE_CHUNK=8192\n'
clocks=dict(q1_clock=clock(get(base,'ATTN_PORT_CUDA_SRC')),partial_clock=clock(get(base,'PARTIAL_CUDA')))
for key,c in clocks.items():(o/(key+'.cu')).write_text(c)
bundle=dict(base=base,adapter=adapter,arms=arms,checker=checker,clocks=clocks,
            CPP=get(base,'ATTN_PORT_CPP_SRC'),partial_CPP=get(base,'PARTIAL_CPP'),VQ8_CPP=get(base,'VQ8_CPP'),profile=(H/'roof_profile.py.inc').read_text())
if args.page_ring:
    bundle['page_clocks']={name:clock(get((o/(name+'.py')).read_text(),'ATTN_PORT_CUDA_SRC')) for name in ('R4','R8')}
blob=base64.b85encode(lzma.compress(json.dumps(bundle,separators=(',',':')).encode(),preset=9)).decode()
body=(H/'roof_lab.py.inc').read_text()
if args.page_ring:
    body=body.replace('1671223+7919*i','2671223+7919*i')
    body=body.replace("order=['A','A2','F','G','P','H1','H2']", "order=['A','A2','G','R4','R8','D']")
    body=body.replace("for name in ('A','F','G','P','H1','H2'):", "for name in ('A','G','R4','R8','D'):")
    body=body.replace("for name in ('A','H1','H2')", "for name in ('A','D')")
    body=body.replace("# Profile the actual S15 N128 code", "# Profile the actual S15 N128 code")
    body=body.replace("    e0,e1=torch.cuda.Event", "    resources['q1']={name:dict(ctas=spaces[name]['_ATTN_PORT_EXT'].occupancy(1),clusters2=spaces[name]['_ATTN_PORT_EXT'].max_cluster(1,2)) for name in ('A','G','R4','R8')}\n    e0,e1=torch.cuda.Event")
    body=body.replace("        exec(compile(_BUNDLE['profile']", "        profile_modules['q1_arms']={'A':(spaces['A'],profile_modules['q1_clock']),**{name:(spaces[name],load(cu,_BUNDLE['CPP'],['attn_forward','occupancy','max_cluster'],['-DFULL_CLOCK=1'])) for name,cu in _BUNDLE['page_clocks'].items()}}\n        exec(compile(_BUNDLE['profile']")
    body=body.replace("counts['official']==224", "counts['official']==192").replace("counts['scored']==336", "counts['scored']==288")
    body=body.replace('H1/H2 one-head residency,F balanced tiles,G quad pages,P native verify1','G quad pages,R4/R8 async page-coordinate lookahead,D tensor-core denominator;TASK gates only')
payload='import ast,base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\n_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+body
(o/('page_attention.py' if args.page_ring else 'roof_attention.py')).write_text(payload)
bootstrap=host(base).replace('# okc: problem=muse_decode_layer','# okc: problem=muse_attention_decode',1)+adapter
(o/'host.py').write_text(bootstrap)
meta=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),arms={n:s['sha256'] for n,s in arms.items()},
    payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),host_sha256=hashlib.sha256(bootstrap.encode()).hexdigest(),
    seeds=[(2671223 if args.page_ring else 1671223)+7919*i for i in range(8)],shapes=6,arms_count=len(arms)+2,scored_TASK=(len(arms)+2)*48,official_TASK=(len(arms)+2)*32)
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
