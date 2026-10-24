"""Generate the full-q8 measurement candidate and a direct layer adapter."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=ROOT/'problems/muse_attention_decode/cand_verify8_flashdecode.py'


def build():
    source=(ROOT/'problems/muse_attention_decode/cand_v2.py').read_text()
    source=source.replace('def kernel_fn(data: Inputs)','def _incumbent_attn_main(data: Inputs)')
    incumbent=(ROOT/'problems/muse_attention_decode/cand_v6_hybrid.py').read_text()
    fa3=incumbent[incumbent.index('_NORMS = {}'):incumbent.index('_EXT = None',incumbent.index('_NORMS = {}'))]
    imports='''
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
'''
    cuda=(HERE/'verify8.cu').read_text()
    cpp='''
#include <cstdint>
void verify8_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,
 int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,
 int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t kt,int64_t stream);
'''
    driver=(HERE/'driver.py.inc').read_text()
    kernel='VERIFY8_CUDA = '+repr(cuda)+'\nVERIFY8_CPP = '+repr(cpp)+'\n'
    source+=imports+'\n'+fa3+'\n'+kernel+driver
    OUT.write_text(source)
    adapter='''"""Full-q8 split attention; promotion requires an H200 win over FA3."""
import torch
from torch.utils.cpp_extension import load_inline
'''+kernel+driver[:driver.index('def kernel_fn')]
    adapter+='''

def attention_verify8(qkv,data,setting=None):
    splits=0
    if isinstance(qkv,tuple):
        raw,count=qkv
        assert raw.is_contiguous()
        if raw.dtype==torch.float32:
            splits=count
        else:
            assert count==1 and raw.dtype==torch.bfloat16
    else:
        raw=qkv
    cfg=dict(data['config'],num_q_heads=16,num_kv_heads=2,head_dim=64)
    return verify8_attention(raw,dict(data,config=cfg),setting,projection_splits=splits).view(cfg['batch']*8,1024)
'''
    (ROOT/'problems/muse_decode_layer/dev/verify8_flashdecode.py').write_text(adapter)
    return OUT


if __name__=='__main__':
    print(build())
