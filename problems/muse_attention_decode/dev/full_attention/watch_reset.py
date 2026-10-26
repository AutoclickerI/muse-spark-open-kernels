"""Wait on filesystem completion events for this lane's five reset batches."""
import ctypes
import json
import os
import select
from pathlib import Path
R=Path(__file__).resolve().parents[4]
TOKENS=('softmax_mix_lab.py','decode_softmax_lab.py','lab_nearby_payload.py','lab_release_payload.py','lab_library_payload.py')
lib=ctypes.CDLL(None,use_errno=True);fd=lib.inotify_init1(os.O_NONBLOCK)
assert fd>=0
for p in [R/'runs/pending',R/'runs/waiting',R/'runs/muse_bf16_proj',R/'runs/muse_decode_layer']:
    assert lib.inotify_add_watch(fd,os.fsencode(p),0x100|0x8|0x80|0x2)>=0
seen=set();finished=set()
def scan():
    ledger={}
    for line in (R/'runs/ledger.jsonl').read_text().splitlines():
        try:
            d=json.loads(line);ledger[d['job']]=d
        except (ValueError,KeyError):continue
    for root in ('pending','muse_bf16_proj','muse_decode_layer'):
        for p in (R/'runs'/root).glob('*.json'):
            try:d=json.loads(p.read_text())
            except (OSError,ValueError):continue
            # The judge rewrites source before computing content_hash. Use
            # accepted job metadata as the bridge to the frozen local wrapper.
            job=d.get('job',d.get('id'))
            d=dict(ledger.get(job,{}),**d)
            token=next((t for t in TOKENS if t in str(d.get('file','')) or t in str(d.get('note',''))),None)
            if token is None:
                aliases={'lab_s15_nearby_repaired.py':TOKENS[2],'lab_s15_release_repaired.py':TOKENS[3],
                         'lab_s15_library_repaired.py':TOKENS[4]}
                token=next((t for alias,t in aliases.items() if alias in str(d.get('file',''))),None)
            if token is None:
                # Completed canonical receipts omit pending metadata. Identity
                # of the frozen wrapper is the stable bridge.
                hashes={'030864131ce8d08f':TOKENS[0],'06e39759a67ca297':TOKENS[1],
                        '5d0893419ae88280':TOKENS[2],'f067eb263e1a67ac':TOKENS[3],'54aa3bedc460bbc8':TOKENS[4],
                        'e5a7d14f2749b3a4':TOKENS[2],'02c19fbd6475cac5':TOKENS[3],'2567c5a90528c832':TOKENS[4]}
                token=next((t for h,t in hashes.items() if str(d.get('content_hash','')).startswith(h)),None)
            if token is None:continue
            key=token,job,root
            if key not in seen:print(root,token,job,str(p.relative_to(R)),flush=True);seen.add(key)
            if root!='pending':finished.add(token)
scan()
while len(finished)<len(TOKENS):
    select.select([fd],[],[])
    try:os.read(fd,1<<20)
    except BlockingIOError:pass
    scan()
os.close(fd)
