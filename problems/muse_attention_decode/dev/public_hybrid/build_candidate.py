import argparse,difflib,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parent/'public_ports';o=H/'out'
a=argparse.ArgumentParser();a.add_argument('--base',type=Path,required=True);a.add_argument('--selection',type=Path,default=H/'selection.json');a.add_argument('--stem',default='candidate');a=a.parse_args();source=a.base.read_text();sel=json.loads(a.selection.read_text());r=json.loads((H/'result.json').read_text());assert r['measured']
admit={(p['cell'],p['arm']) for p in r['points'] if p['nominated']};assert all((c,v) in admit for c,v in sel.items());dispatch={};sources={}
for c,arm in sel.items():
 b,l=map(int,c[:-1].split('x'));w=c[-1]=='s';dispatch[b,l,w]=arm
 n='operator_direct' if w and (arm=='O' or b==1 and arm!='J') else 'lee_hybrid';sources[n]=(o/('operator_selected.py' if n=='operator_direct' else 'lee_selected.py')).read_text()
s=source+'\n_HY_PUBLIC_NOTICES='+repr((P/'NOTICES.txt').read_text())+'\n_HY_SELECTION='+repr(dispatch)+'\n_HY_SOURCES='+repr({n:sources[n] for n in sorted(sources)})+'\n'+(H/'runtime.py.inc').read_text();compile(s,'hybrid_candidate','exec');(o/(a.stem+'.py')).write_text(s)
path=str(a.base.resolve().relative_to(R));patch=''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='a/'+path,tofile='b/'+path));(H/(a.stem+'.patch')).write_text(patch)
m=dict(base=path,base_sha256=hashlib.sha256(source.encode()).hexdigest(),source_sha256=hashlib.sha256(s.encode()).hexdigest(),patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),selection=sel,modules=list(sorted(sources)),scope='Only measured selected attention cells; original S21/S22 dispatch elsewhere. Entire prepare and held-out G0 remain P4 confirmation.')
(H/(a.stem+'_manifest.json')).write_text(json.dumps(m,indent=2)+'\n');print(m)
