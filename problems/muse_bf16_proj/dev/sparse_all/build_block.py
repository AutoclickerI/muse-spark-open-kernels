"""Replace only scalar point loads with a shared-codeword tile reader."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;s=(H/'candidate.py').read_text();body=(H/'triton.py.inc').read_text()
assert body.count('for kb in tl.range')==3 and body.count('for kk in tl.range')==1
body=body.replace('    for kb in tl.range','    ctx=dc_block_context(W)\n    for kb in tl.range').replace('    for kk in tl.range','    ctx=dc_block_context(W)\n    for kk in tl.range').replace('    for k in range(k0','    ctx=dc_block_context(W)\n    for k in range(k0')
body=body.replace('dc_load(W,n[:,None],(k+kk)[None,:],K)','dc_block_load(ctx,pn*BN,k,K,BN,BK)')
body=body.replace('dc_load(W,n[:,None],(kb*BK+kk)[None,:],K)','dc_block_load(ctx,pn*BN,kb*BK,K,BN,BK)')
body=body.replace('dc_load(W,rw[:,None],(kk*BK+rk)[None,:],K)','dc_block_load(ctx,pw*BW,kk*BK,K,BW,BK)')
assert 'dc_load(' not in body
delta=(H/'block_reader.py.inc').read_text()+'\n'+body;out=s+'\n'+delta;compile(out,'<block candidate>','exec');assert len(out.encode())<1000000
(H/'block_delta.py.inc').write_text(delta);(H/'block_candidate.py').write_text(out)
(H/'block_manifest.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),delta_sha256=hashlib.sha256(delta.encode()).hexdigest(),bytes=len(out.encode())),indent=2)+'\n');print(len(out),hashlib.sha256(out.encode()).hexdigest()[:8])
