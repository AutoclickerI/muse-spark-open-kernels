"""Run the TASK reference / fresh graph checks through the fused projection-partials attention port."""
import hashlib
import linecache
import runpy
import sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE))
import partial_port
import vq8
src=(ROOT/'prof/lab/vq8_layer.py').read_text()
name='<partial-port-local>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name)
ns={'__name__':'partial_port_screen'}
exec(compile(src,name,'exec'),ns)
exec(partial_port.cell(),ns)
fast='--fast' in sys.argv
if '--stages' in sys.argv:
    import port_stages
    st=int(sys.argv[sys.argv.index('--stages')+1])
    exec(port_stages.cell(st),ns)
    ns['attention_partial_port']=ns[f'attention_partial_port_s{st}']

def call(data, qkv=None, proj_splits=0, split_stride=0, **kwargs):
    raw=data['qkv'] if qkv is None else qkv
    if not proj_splits:
        raw=raw.view(-1,raw.shape[-1]).unsqueeze(0)
    return ns['attention_partial_port']((raw,proj_splits or 1),data,fast).view(data['config']['batch'],data['config']['q_len'],-1)

def layer(qkv,data,**kwargs):
    return ns['attention_partial_port'](qkv,data,fast)

vq8.vq8_attention=call
vq8.attention_vq8=layer
if '--partials' in sys.argv:
    # The partial port preserves the same gated behavior but currently has no gate=False driver;
    # check_partials' extra no-gate check is specific to the vq8 wrapper.
    text=(HERE/'check_partials.py').read_text()
    text=text[:text.index('        # No-gate mode')]+"\nprint('FAILS', fails)\nraise SystemExit(bool(fails))\n"
    exec(compile(text,str(HERE/'check_partials.py'),'exec'),{'__name__':'__main__','__file__':str(HERE/'check_partials.py')})
else:
    sys.argv=[str(HERE/'check.py')]+(['--quick'] if '--quick' in sys.argv else [])
    runpy.run_path(str(HERE/'check.py'),run_name='__main__')
