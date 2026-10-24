"""Build the self-contained sliding attention candidate and layer adapter."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def build():
    base = (ROOT / 'problems/muse_attention_decode/cand_v2.py').read_text()
    base = base.replace('def kernel_fn(data: Inputs)', 'def _full_attn_main(data: Inputs)')
    other = (ROOT / 'problems/muse_attention_decode/cand_v6_hybrid.py').read_text()
    fa3 = other[other.index('_NORMS = {}'):other.index('_EXT = None', other.index('_NORMS = {}'))]
    imports = '''
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate
'''
    # Reuse the proven MMA fragments, with M's eight rows now indexing query tokens.
    cuda = base[base.index('CUDA_SRC = r"""')+len('CUDA_SRC = r"""'):base.index('\n"""',base.index('CUDA_SRC = r"""'))]
    a = cuda.index('DEVI uint32_t smem_u32')
    z = cuda.index('template <int L, int STAGES>\n__global__')
    helpers = cuda[a:z]
    helpers = helpers.replace('return (((t >> 1) & 3) << 1) ^ ((t & 1) << 2);','return t & 3;')
    helpers = helpers.replace('bool need_mask, float scale_log2)', 'bool need_mask, float scale_log2, int nq, bool same_query)')
    helpers = helpers.replace('const int j = (L == 1) ? 0 : (2 * (2 * wh + mt) + (e >> 1));', 'const int j = same_query ? 0 : g;')
    helpers = helpers.replace('if (kp < lo || kp >= hi)', 'if (kp < lo || kp >= hi || j >= nq)')
    prefix = '''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <stdexcept>
#define DEVI __device__ __forceinline__
namespace {
constexpr int ROWB=64;
'''
    cuda = prefix+helpers+(HERE/'mma_body.cu.inc').read_text()
    cpp = '''
#include <cstdint>
void window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,
 int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,
 int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,
 int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream);
'''
    driver = (HERE / 'driver.py.inc').read_text()
    source = base + imports + '\n' + fa3 + '\nWINDOW_CUDA = ' + repr(cuda) + '\nWINDOW_CPP = ' + repr(cpp) + '\n' + driver
    out = ROOT / 'problems/muse_attention_decode/cand_sliding_singlelaunch.py'
    out.write_text(source)
    # Dense-layer config has fixed TP8 head counts and the projected QKV is separate.
    adapter = '"""One-launch sliding attention after the layer QKV projection. Await H200 promotion."""\n'
    adapter += 'import torch\nfrom torch.utils.cpp_extension import load_inline\n\n_INV_FREQ = {}\n\n'
    inv = base[base.index('def _inv_freq'):base.index('\n\n_SMS =')]
    adapter += inv + '\n\nWINDOW_CUDA = ' + repr(cuda) + '\nWINDOW_CPP = ' + repr(cpp) + '\n'
    adapter += driver[:driver.index('def kernel_fn')].replace('_counters(B*HQ, raw.device)', '_window_counters(B*HQ, raw.device)')
    adapter += '''
_WINDOW_COUNTERS = {}
def _window_counters(n, device):
    key = str(device)
    if key not in _WINDOW_COUNTERS:
        _WINDOW_COUNTERS[key] = torch.zeros(1 << 16, dtype=torch.int32, device=device)
    assert n <= (1 << 16)
    return _WINDOW_COUNTERS[key]


def attention_sliding(raw, data):
    splits = 0
    if isinstance(raw, tuple):
        partial, count = raw
        assert partial.is_contiguous()
        if partial.dtype == torch.float32:
            splits = count
        else:
            assert partial.dtype == torch.bfloat16 and count == 1
        raw = partial
    cfg = dict(data['config'], num_q_heads=16, num_kv_heads=2)
    metadata = dict(data, config=cfg)
    return sliding_attention(raw, metadata, splits=splits).view(cfg['batch']*cfg['q_len'], 1024)


if torch.cuda.is_available():
    _window_counters(1 << 16, torch.device('cuda', torch.cuda.current_device()))
'''
    (ROOT/'problems/muse_decode_layer/dev/sliding_attention.py').write_text(adapter)
    return out

if __name__ == '__main__':
    print(build())
