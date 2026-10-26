"""Prepare the vector-decoder follow-up; leave the already queued source frozen."""
from pathlib import Path
import ast,runpy,hashlib,json,sys
H=Path(__file__).resolve().parent
f=runpy.run_path(str(H/'fast_decoder.py'));s=(H/'candidate.py').read_text();view='--view' in sys.argv
node=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='DC_NATIVE' for t in n.targets))
native=ast.literal_eval(node.value);lines=s.splitlines(keepends=True)
changed=f['view_source' if view else 'vector_source'](native)
lines[node.lineno-1:node.end_lineno]=['DC_NATIVE = '+repr(changed)+'\n']
out=''.join(lines)
if view:out+='\n'+(H/'fast_decoder.py').read_text()+'\ninstall_view(globals())\n'
compile(out,'<decoder candidate>','exec');assert len(out.encode())<1000000
label='view' if view else 'vector';(H/(label+'_candidate.py')).write_text(out)
(H/(label+'_manifest.json')).write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),
    base_candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),native_sha256=hashlib.sha256(changed.encode()).hexdigest(),bytes=len(out.encode())),indent=2)+'\n')
print(label,len(out),hashlib.sha256(out.encode()).hexdigest()[:8])
