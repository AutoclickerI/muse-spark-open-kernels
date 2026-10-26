"""Build synthetic lab sources from tracked public files, without reference imports."""
import ast
import base64
import hashlib
import lzma
import re


def public_bundle(problem):
    source=(problem/'reference.py').read_text();chunks=['import torch, math, functools']
    names={'_stream_basis','_longest','_lengths','_quant','_rms','_fill_cache','_fa3_scheduler_metadata','generate_input'}
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign) and (n.lineno<89 or 150<n.lineno<202):chunks.append(ast.get_source_segment(source,n))
        elif isinstance(n,ast.FunctionDef) and n.name in names:
            chunks.append(('@functools.cache\n' if n.name=='_stream_basis' else '')+ast.get_source_segment(source,n))
    inputs='\n\n'.join(chunks)+'\nCACHE_CHUNK=8192\n'
    reference='\n'.join(l for l in source.splitlines() if not l.startswith(('from checks import','from nvfp4 import')))
    checker='\n'.join(l for src in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),reference)
                      for l in src.splitlines() if not l.startswith('from __future__ import'))
    return dict(arms={},inputs=inputs,checker=checker)


def host(source):
    digest=hashlib.sha256(source.encode()).hexdigest();assert digest.startswith('60254883')
    notice=re.search(r'(?m)^# MIT License\n(?:#.*\n)*',source).group()
    encoded=base64.b85encode(lzma.compress(source.encode(),preset=9)).decode()
    return '# okc: problem=muse_decode_layer\n# Nathen Fernandes; MIT\n'+notice+f'''import base64,lzma,linecache,hashlib,torch
_s15_source=lzma.decompress(base64.b85decode({encoded!r})).decode()
assert hashlib.sha256(_s15_source.encode()).hexdigest()=={digest!r}
_s15_path="<s15-expanded-602548834f74>"
linecache.cache[_s15_path]=(len(_s15_source),None,_s15_source.splitlines(True),_s15_path)
exec(compile(_s15_source,_s15_path,"exec"),globals())
_s15_prepare=globals()["prepare"]
def prepare(weights):
    return _s15_prepare(weights)

_s15_entry_kernel=globals()["kernel_fn"]
def kernel_fn(data,state):
    return _s15_entry_kernel(data,state)
'''
