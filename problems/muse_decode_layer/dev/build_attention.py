from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
s=(root/'cand_moe4.py').read_text()
a=(root/'cand_adapt4.py').read_text()
custom=a[a.index('CUDA_SRC ='):a.index('\n\ndef stage_attention',a.index('CUDA_SRC ='))]
custom=custom.replace('const __nv_bfloat16* qkv;', 'const void* qkv;\n  int qkv_splits, qkv_ps, qkv_float;')
# Consume the GEMV partials in attention itself, preserving the BF16 activation boundary.
needle='DEVI uint32_t smem_u32'
helper='''DEVI __nv_bfloat162 qkv_pair(const Params& p, long long off) {
  if (!p.qkv_float) return *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(p.qkv) + off);
  float a = 0.f, b = 0.f;
  for (int s = 0; s < p.qkv_splits; ++s) {
    float2 z = *reinterpret_cast<const float2*>(static_cast<const float*>(p.qkv) + (long long)s * p.qkv_ps + off);
    a += z.x; b += z.y;
  }
  return __floats2bfloat162_rn(a, b);
}

'''
custom=custom.replace(needle,helper+needle)
custom=re.sub(r'\*reinterpret_cast<const __nv_bfloat162\*>\(\s*p.qkv \+ (.*?)\);',r'qkv_pair(p, \1);',custom,flags=re.S)
custom=custom.replace('int64_t qkv, int64_t positions,', 'int64_t qkv, int64_t qkv_splits, int64_t qkv_ps, int64_t qkv_float, int64_t positions,')
custom=custom.replace('p.qkv = reinterpret_cast<const __nv_bfloat16*>(qkv);','p.qkv = reinterpret_cast<const void*>(qkv);\n  p.qkv_splits = qkv_splits; p.qkv_ps = qkv_ps; p.qkv_float = qkv_float;')
custom=custom.replace('    cfg = data["config"]\n    batch, q_len', '    qkv, splits = qkv\n    cfg = data["config"]\n    batch, q_len')
custom=custom.replace('qkv.data_ptr(), data["positions"]', 'qkv.data_ptr(), splits, qkv.stride(0), int(qkv.dtype == torch.float32), data["positions"]')
custom=custom.replace('    return out.view(-1, hq * d)', '    return out.view(-1, hq * d)')
# Hash the modified source so kernels cannot collide with the old extension.
custom=custom.replace('name=f"muse_decode_attn_sm{arch}"', 'name=f"muse_decode_attn_split_{arch}"')
# Existing helper routines use cpp_sources and cuda_sources from this module.
s=s.replace('import torch\n', 'import torch\nfrom torch.utils.cpp_extension import load_inline\n',1)
s+='\n\n'+custom
s+='''

_attention_fallback = stage_attention

def stage_attention(qkv, data, state):
    # Portable CUDA path is deliberately tested locally too.
    cfg = data["config"]
    if cfg["q_len"] == 1 and cfg["batch"] > 1:
        return _layer_attention_custom(qkv, data)
    return _attention_fallback(qkv, data, state)
'''
(root/'cand_moe4_attn.py').write_text(s)
