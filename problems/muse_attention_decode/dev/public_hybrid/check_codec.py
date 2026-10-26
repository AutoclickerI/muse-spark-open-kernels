"""Round-trip actual and adversarial timing streams before using judge output."""
import base64,copy,hashlib,json,lzma,random,math
from pathlib import Path
H=Path(__file__).resolve().parent;exec((H/'codec.py.inc').read_text());results=[]
for p in (H/'out/normal.json',H/'out/forced.json',H.parent/'public_ports/receipt_data.json'):
 original=json.loads(p.read_text());encoded=encode_rounds(copy.deepcopy(original));decoded=decode_rounds(copy.deepcopy(encoded));assert decoded==original
 wire=base64.b85encode(lzma.compress(json.dumps(encoded,separators=(',',':')).encode())).decode();results.append(dict(source=str(p),elements=sum(len(v) for samples in original['round_ns'].values() for sample in samples for v in sample.values()),wire_bytes=len(wire),sha256=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()))
# Full planned extent with large native-scale jitter and every nanosecond retained.
meta=json.loads((H/'out/lab_manifest.json').read_text());rng=random.Random(912733);rows={}
for cell,names in meta['case_arms'].items():
 rows[cell]=[{n:[rng.randrange(5000,330000)+rng.randrange(12000) for _ in range(2*len(names))] for n in names} for _ in meta['seeds']]
x=dict(round_ns=rows,seeds=meta['seeds']);assert decode_rounds(encode_rounds(copy.deepcopy(x)))==x
results.append(dict(adversarial=True,exact=True));(H/'codec_validation.json').write_text(json.dumps(results,indent=2)+'\n');print(results)
