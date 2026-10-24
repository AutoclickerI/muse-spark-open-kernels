"""Screen the serving scale-layout migration through the standard MoE tests/graphs."""
# okc files must start with their header; this builder writes that adapter separately.
from pathlib import Path
p=Path('serving/sglang/python/sglang/srt/layers/muse_spark_moe.py')
s=p.read_text();s+='''
_serving_prepare=prepare

def prepare(weights):
    w=dict(weights,w13_gs_gate=weights['w13_gs'],w13_gs_up=weights['w13_gs'])
    return _serving_prepare(w)
'''
Path('prof/lab/_serving_raw_scale_screen.py').write_text('# okc: problem=muse_moe_w4a16\n'+s)
