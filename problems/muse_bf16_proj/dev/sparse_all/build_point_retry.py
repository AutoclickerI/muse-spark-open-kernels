"""Remeasure only the previously ungated grouped section, no lost native reruns."""
from pathlib import Path
import base64,zlib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[3]
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
encoded=base64.b64encode(zlib.compress((H/'candidate.py').read_bytes(),9)).decode()
s=notice+'\nimport torch,base64,zlib\nDC_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\nBLOCK_SOURCE='+repr((H/'block_delta.py.inc').read_text())+'\n'
s+=(H/'decoder_lab.py.inc').read_text().split('def lab(deadline')[0]+'\n'+(H/'point_lab.py.inc').read_text()+'''
def lab(deadline,weights,state=None):
    import math,time
    m=dl_namespace();T=torch;gen=T.Generator(device='cuda').manual_seed(61791)
    def dense(n,k):return (T.randn((n,k),device='cuda',generator=gen)/math.sqrt(k)).bfloat16()
    if weights is None or 'pooler' not in weights:
        weights=dict(qkv_gate=dense(2304,8192),o=dense(8192,1024),pooler=dense(4096,8192),unpooler=dense(8192,4096))
        mean=T.randn(8192,generator=T.Generator().manual_seed(5354)).cuda()
        weights['router']=.045*T.randn((256,8192),device='cuda',generator=gen)+.07*T.randn((256,1),device='cuda',generator=gen)*mean/math.sqrt(8192)
    roles=m['dc_role_weights'](weights);ctrl=dict(roles);m['_seed_v9_gemv_choices'](ctrl);m['cd_module'](((1,1,2,3),));m['_cd_prepare'](ctrl)
    ctrl['pre_expert_w']=T.ones(4096,device='cuda')
    # Published router-bias distribution; zero bias artificially creates equal
    # FP32 sigmoid values at large M. Exact membership remains a required gate.
    ctrl['router_bias']=weights.get('router_bias',.055*T.randn((256,),device='cuda',generator=T.Generator(device='cuda').manual_seed(81901)))
    banks={n:m['dc_encode_cd'](ctrl['coded_dense'][n])[0] for n in ('qkv_gate','pool_router')}
    return point_lab(deadline,(m,weights,roles,ctrl,banks))
'''
compile(s,'<point retry>','exec');(H/'p10_point_retry.py').write_text(s)
(H/'point_retry_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),cells=20,seeds=8),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
