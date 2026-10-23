"""Build v3's token-tiled pipeline with a direct reader of the single canonical V6 bank."""
from pathlib import Path
import re
p=Path(__file__).resolve().parents[1]
s=(p/'dev/moe_w4a16_fast.cu').read_text()
s=s.replace('constexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;', 'constexpr int STAGE_BYTES = X_BYTES;')
s=s.replace('  const uint8_t* sfblob;','  const float* gs_up;')
s=s.replace('      const uint8_t* wsrc = p.wblob + ((size_t)(e * p.nfb + fb) * KT) * W_BYTES;\n      const uint8_t* ssrc = p.sfblob + ((size_t)(e * p.nfb + fb) * KT) * SF_BYTES;\n','')
s=s.replace('        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);\n','')
a=s.index('    auto load_words ='); b=s.index('    auto issue =',a)
s=s[:a]+'''    // V6 m16 tiles hold paired gate/up rows (or 16 down rows). No converted bank.
    auto load_words = [&](int k) {
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        constexpr int MTILES = MODE ? 512 : 256;
        const uint8_t* wt = p.wblob + ((size_t)e * MTILES + fb*16 + cw*8 + j*4 + warp)*36864;
        raw[j] = *reinterpret_cast<const uint4*>(wt + (k*32 + lane)*16);
        uint16_t sc = *reinterpret_cast<const uint16_t*>(wt + 32768 + (k/8)*512 + lane*16 + (k%8)*2);
        // Every k16 instruction uses the same two row scales after V6's K permutation.
        sfw[j].x = uint32_t(sc) | (uint32_t(sc)<<16);
        sfw[j].y = sfw[j].x;
      }
    };
'''+s[b:]
s=s.replace('load_words(s);','load_words(0);',1)
s=s.replace('load_words(s);','load_words(k + 1);')
s=s.replace('const float gsc = p.gs[e] * 256.f;','const float gsc = p.gs[e] * 256.f;\n    const float usc = p.gs_up[e] * 256.f;')
a=s.index('    const int r0 = warp * 16');b=s.index('    fence_async_smem();',a)
s=s[:a]+'''    const int c2 = 2 * (lane & 3);
    constexpr int NT = MODE == 1 ? 1 : 2;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      uint8_t* tile = my_epi + (MODE ? 0 : j * X_BYTES);
#pragma unroll
      for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
        for (int h = 0; h < (MODE ? 1 : 2); ++h) {
          const int rho = MODE ? j*32 + warp*8 + (lane>>2) : warp*16 + (lane>>2) + h*8;
          // Natural feature -> V6 activation-K pair order, inverse of load_x's permutation.
          const int pair = rho >> 1;
          const int rp = 2*((pair>>3) + ((pair&1)<<2) + ((pair&6)<<2)) + (rho&1);
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int n = 8 * i + c2 + q;
            float v;
            if constexpr (MODE == 1) {
              float gg = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+q]*gsc));
              float uu = __bfloat162float(__float2bfloat16_rn(acc[j][4*i+2+q]*usc));
              v = gg / (1.f + __expf(-gg)) * uu;
            } else {
              v = acc[j][4*i+2*h+q]*gsc;
            }
            *reinterpret_cast<__nv_bfloat16*>(tile+n*128+(((rp>>3)^(n&7))<<4)+(rp&7)*2) = __float2bfloat16_rn(v);
          }
        }
      }
    }
'''+s[b:]
s=s.replace('void moe_gemm(torch::Tensor wblob, torch::Tensor sfblob, torch::Tensor gs, torch::Tensor x, torch::Tensor y,','void moe_gemm(torch::Tensor wblob, torch::Tensor gs, torch::Tensor gs_up, torch::Tensor x, torch::Tensor y,')
s=s.replace('  p.sfblob = sfblob.data_ptr<uint8_t>();','  p.gs_up = gs_up.data_ptr<float>();')
a=s.index('// Grouped NVFP4');b=s.index('#include <cuda.h>',a)
s=s[:a]+'''// Token-tiled, warp-specialized WGMMA reader of canonical V6 m16 tiles.
// Each V6 tile is 32768 packed-code bytes followed by 4096 encoded-scale bytes.
// W13 rows are eight gate + eight up; W2 rows are sixteen natural output rows.
// Activations and outputs use the inverse V6 K permutation inside 128B-swizzled
// [token_tile, k64, token128, k64] buffers. Weights load directly into registers;
// only activation tiles are staged by the producer. Separate gate/up FP32 scales.
'''+s[b:]
s=s.replace('constexpr int W_BYTES = 4 * 2048;    // 8 KB\nconstexpr int SF_BYTES = 4 * 256;    // 1 KB\n','')
s=s.replace('constexpr int STAGE_BYTES = X_BYTES;   // 25600 (multiple of 1024)','constexpr int STAGE_BYTES = X_BYTES;')
s='// Direct canonical V6 reader; bank contract docs/reference/expert_layout.md.\n'+s
(p/'dev/moe_w4a16_canonical.cu').write_text(s)
print(p/'dev/moe_w4a16_canonical.cu')
