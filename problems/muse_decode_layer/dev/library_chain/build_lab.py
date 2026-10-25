"""Single paired full-layer payload: L, per-replacement leave-one-out and A:A."""
import ast
import base64
import difflib
import hashlib
import json
import lzma
from pathlib import Path
import build
H=Path(__file__).resolve().parent;P=H.parents[1]


def main():
    base=build.BASE.read_text();before=base.splitlines(True);specs={}
    for name,without in [('L',())]+[('no_'+g,(g,)) for g in build.GROUPS]:
        src=build.compose(without);after=src.splitlines(True)
        edits=[(a,b,''.join(after[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
        specs[name]=dict(edits=edits,sha256=hashlib.sha256(src.encode()).hexdigest())
    # Generator uses the same public model distribution and random permutation;
    # setup chunk changes scratch capacity only, all keys remain in the cache.
    gen=H.parent/'nearby/inputs.py'
    if not gen.exists():raise RuntimeError('run nearby/build.py once for the frozen public input generator')
    inputs=gen.read_text()
    # Private independent reference for this synthetic lab only. Never import
    # the judge reference module, and never use its results in timed kernel_fn.
    # Preserve the public checker, thresholds and arithmetic byte for byte.
    problem=P
    check=(problem/'checks.py').read_text()
    nvfp4=(problem/'nvfp4.py').read_text()
    reference=(problem/'reference.py').read_text()
    reference='\n'.join(l for l in reference.splitlines() if not l.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(l for src in (check,nvfp4,reference) for l in src.splitlines()
                      if not l.startswith('from __future__ import'))
    bundle=json.dumps(dict(arms=specs,inputs=inputs,checker=checker),separators=(',',':'))
    enc=base64.b85encode(lzma.compress(bundle.encode(),preset=9)).decode()
    text='import torch,time,statistics,math,json,gc,ast,hashlib,linecache,types,base64,lzma\n'
    text+='_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(enc)+')))\n'
    text+=(H/'lab.py.inc').read_text()
    out=H/'out';out.mkdir(exist_ok=True);(out/'library_payload.py').write_text(text)
    host=(P/'dev/s15/s15ab_host.py').read_text();(out/'library_host.py').write_text(host)
    meta=dict(arms={k:v['sha256'] for k,v in specs.items()},seeds=8,replays=8,shapes=12,loads=6,
              base_sha256=build.SHA,payload_sha256=hashlib.sha256(text.encode()).hexdigest(),
              host_sha256=hashlib.sha256(host.encode()).hexdigest(),
              checker_sha256=hashlib.sha256(checker.encode()).hexdigest())
    (H/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
