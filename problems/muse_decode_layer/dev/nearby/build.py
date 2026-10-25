"""Freeze S15, public input generator and SGLang baseline in one decode lab."""
import ast
import base64
import hashlib
import difflib
import json
import lzma
from pathlib import Path
from cases import SEEDS, S15_SHA, cases

H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    p=R/'problems/muse_decode_layer';candidate=p/'cand_r3_best7_s15.py'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest()==S15_SHA
    source=(p/'reference.py').read_text();tree=ast.parse(source)
    names={'_stream_basis','_longest','_lengths','_quant','_rms','_fill_cache',
           '_fa3_scheduler_metadata','generate_input'}
    chunks=['import torch, math, functools']
    for node in tree.body:
        if isinstance(node,ast.Assign) and (node.lineno<89 or 150<node.lineno<202):
            chunks.append(ast.get_source_segment(source,node))
        elif isinstance(node,ast.FunctionDef) and node.name in names:
            # ast.get_source_segment omits decorators: preserve stream-basis caching.
            chunks.append(('@functools.cache\n' if node.name=='_stream_basis' else '')+ast.get_source_segment(source,node))
    inputs='\n\n'.join(chunks)+'\nCACHE_CHUNK=8192\n'
    # Readable generated artifact for the local checker; none of the official
    # reference/checking module is imported by a remote candidate.
    (H/'inputs.py').write_text(inputs)
    baseline=(p/'baseline.py').read_text()
    enc=lambda s:base64.b85encode(lzma.compress(s.encode(),preset=9)).decode()
    payload='import torch,time,statistics,math,json,gc,base64,lzma,linecache,types\n'
    payload+='CASES='+repr(cases())+'\nSEEDS='+repr(SEEDS)+'\nS15_SHA='+repr(S15_SHA)+'\n'
    nsource=(p/'cand_r3_best7_s15n.py').read_text()
    before=candidate.read_text().splitlines(True);after=nsource.splitlines(True)
    edits=[(a,b,''.join(after[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
    payload+='S15N_SHA='+repr(hashlib.sha256(nsource.encode()).hexdigest())+'\nS15N_EDITS='+repr(edits)+'\n'
    payload+='INPUT_SOURCE=lzma.decompress(base64.b85decode('+repr(enc(inputs))+')).decode()\n'
    payload+='BASELINE_SOURCE=lzma.decompress(base64.b85decode('+repr(enc(baseline))+')).decode()\n'
    payload+=(H/'lab.py.inc').read_text()
    (H/'nearby_payload.py').write_text(payload)
    # Compress only for upload size; the expanded host is byte-for-byte S15.
    # Plain MIT notice is preserved in this known screened host bootstrap.
    host=(p/'dev/s15/s15ab_host.py').read_text()
    assert S15_SHA in host
    (H/'nearby_host.py').write_text(host)
    meta=dict(candidate_sha256=S15_SHA,s15n_sha256=hashlib.sha256(nsource.encode()).hexdigest(),cases=len(cases()),nearby=len(cases())-12,
              seeds=SEEDS,replays=8,cold_l2='1GiB read before each side',
              source_sha256={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in (candidate,p/'reference.py',p/'baseline.py',H/'nearby_host.py',H/'nearby_payload.py')})
    (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))


if __name__=='__main__':main()
