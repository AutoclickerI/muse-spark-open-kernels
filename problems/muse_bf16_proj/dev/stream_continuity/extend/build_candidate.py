"""Pack the frozen ALL source for the regular local_eval entry point."""
import ast,base64,hashlib,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
s=(H/'ALL_trial.py').read_text();sha=hashlib.sha256(s.encode()).hexdigest()
old=(R/'problems/muse_decode_layer/cand_r3_best7_s18_final.py').read_text()
head='\n'.join(l.rstrip() for l in old.splitlines()[:ast.parse(old).body[0].lineno-1])+'\n'
blob=base64.b85encode(lzma.compress(s.encode(),preset=9)).decode()
packed=head+'import torch,base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=='+repr(sha)+'\n_file="<p7-ALL-'+sha[:12]+'>"\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,"exec"),globals())\n'
(H/'out/ALL_candidate.py').write_text(packed);print(sha)
