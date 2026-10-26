"""Lossless single-file shipment for an expanded, separately audited S16 source."""
import base64,hashlib,lzma
from pathlib import Path
H=Path(__file__).resolve().parent


def pack(source):
    # Always use one representation; its expanded CUDA/precision checks are
    # recorded separately. MIT permission remains visible in the shipped file.
    notice='\n'.join('# '+line for line in (H.parent/'s15/proj/LICENSE.NathenFernandes').read_text().splitlines())
    encoded=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
    sha=hashlib.sha256(source.encode()).hexdigest()
    out='# okc: problem=muse_decode_layer\n'+notice+'\nimport torch,base64,lzma,linecache,hashlib\n'
    out+='_source=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    out+='assert hashlib.sha256(_source.encode()).hexdigest()=='+repr(sha)+'\n'
    out+='_file="<s16-'+sha[:12]+'>"\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\n'
    out+='exec(compile(_source,_file,"exec"),globals())\n'
    out+='_s16_entry_prepare=globals()["prepare"]\n_s16_entry_kernel=globals()["kernel_fn"]\ndef prepare(weights):\n    return _s16_entry_prepare(weights)\ndef kernel_fn(data,state):\n    return _s16_entry_kernel(data,state)\n'
    assert len(out.encode())<1_000_000
    return out
