from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(HERE/'projection.py').read_text()+'\n'+(HERE/'driver.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
out=ROOT/'prof/lab/verify_dense_library.py';out.write_text(s)
(HERE/'manifest.json').write_text(json.dumps(dict(payload=str(out.relative_to(ROOT)),payload_sha256=hashlib.sha256(s.encode()).hexdigest(),projection_sha256=hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest(),host='muse_moe_w4a16',enabled={}),indent=2)+'\n');print(hashlib.sha256(s.encode()).hexdigest())
