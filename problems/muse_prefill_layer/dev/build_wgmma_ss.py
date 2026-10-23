"""Generate shared/shared WGMMA expert path without changing accepted candidate sources."""
from pathlib import Path
import re
p=Path(__file__).resolve().parents[1]
s=(p/'dev/moe_w4a16.cu').read_text()
# Double-buffer decoded m64xk64 matrices for the two consumer WGs.
s=s.replace('uint8_t* epi = smem + STAGES * STAGE_BYTES;', '''constexpr int DEQ_BYTES = MOE_WGMMA ? 65536 : (MODE == 1 ? 32768 : 0);
  uint8_t* deq = smem + STAGES * STAGE_BYTES;
  uint8_t* epi = deq + DEQ_BYTES;''')
# Build the SS instruction from the reviewed RS instruction with the same accumulator order.
i=s.index('  asm volatile(\n',s.index('void mma_block('));j=s.index('\n#else',i)
asm=s[i:j]
asm=asm.replace('%69','%66').replace('"{%64, %65, %66, %67}, %68, p, 1, 1, 0;', '"%64, %65, p, 1, 1, 0, 0;')
asm=asm.replace(': "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));', ': "l"(da), "l"(db), "r"(scale_d));')
ss='''
// A and B are both row/K-major 128B-swizzled shared matrices.
__device__ __forceinline__ void mma_ss(float (&d)[64], const uint8_t* as, const uint8_t* xs, int b, int scale_d) {
#if MOE_WGMMA
  uint64_t da = make_desc_b(smem_u32(as)) + (uint64_t)((b * 32) >> 4);
  uint64_t db = make_desc_b(smem_u32(xs)) + (uint64_t)((b * 32) >> 4);
'''+asm+'''
#else
  const int t = threadIdx.x % 128;
  const int r = (t >> 5) * 16 + ((t & 31) >> 2);
  const int kb = b * 32 + 4 * (t & 3);
  uint32_t a[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int rr = r + 8 * (j & 1), kk = kb + 16 * (j >> 1);
    a[j] = *reinterpret_cast<const uint32_t*>(as + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15));
  }
  mma_block(d, a, xs, b, scale_d);
#endif
}
'''
s=s.replace('\nstruct Params {',ss+'\nstruct Params {')
i=s.index('    // prologue: first k-tile');j=s.index('\n    // ---- epilogue',i)
old=s[i:j]
new='''    if constexpr (MOE_WGMMA || MODE == 1) {
      // Shared A fragments stay alive until the committed group has completed.
      // Hopper has two A slots, so dequantization of the next tile overlaps MMA.
      for (int k = 0; k < KT; ++k) {
        mbar_wait(&full[s], phase);
        load_words(s);
        uint8_t* abase = deq + (MOE_WGMMA ? (k & 1) * 32768 : 0) + cw * 16384;
        const int r = warp * 16 + (lane >> 2);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            uint32_t words[4] = {raw[j].x, raw[j].y, raw[j].z, raw[j].w};
            uint32_t scales[2] = {sfw[j].x, sfw[j].y};
            uint32_t a[4];
            make_frag(words[b], e4m3x2_to_bf16x2(scales[b >> 1] >> (16 * (b & 1))), lo_a, hi_a, a);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int rr = r + 8 * (q & 1), kk = b * 32 + 4 * (lane & 3) + 16 * (q >> 1);
              *reinterpret_cast<uint32_t*>(abase + j * 8192 + rr * 128 + (((kk >> 4) ^ (rr & 7)) << 4) + (kk & 15)) = a[q];
            }
          }
        }
        fence_async_smem();
        named_bar(1 + cw, 128);
        fence_acc(acc[0]); fence_acc(acc[1]); wg_fence();
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          mma_ss(acc[0], abase, smem + s * STAGE_BYTES, b, k > 0 || b > 0);
          mma_ss(acc[1], abase + 8192, smem + s * STAGE_BYTES, b, k > 0 || b > 0);
        }
        wg_commit();
        wg_wait<1>();
        if (MOE_WGMMA && k > 0) {
          __syncwarp();
          if (lane == 0) mbar_arrive(&empty[(s + STAGES - 1) % STAGES]);
        }
        if (!MOE_WGMMA) {
          named_bar(1 + cw, 128);
          if (lane == 0) mbar_arrive(&empty[s]);
        }
        if (k + 1 < KT) {
          if (++s == STAGES) { s = 0; phase ^= 1; }
        }
      }
      wg_wait<0>();
      fence_acc(acc[0]); fence_acc(acc[1]);
      __syncwarp();
      if (MOE_WGMMA && lane == 0) mbar_arrive(&empty[s]);
      if (++s == STAGES) { s = 0; phase ^= 1; }
    } else {
'''+old+'''
    }
'''
s=s[:i]+new+s[j:]
s=s.replace('moe::smem_bytes<M, S>()','(moe::smem_bytes<M, S>() + (major == 9 ? 65536 : (M == 1 ? 32768 : 0)))')
s=s.replace('MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5)', 'MOE_LAUNCH(1, 4); else MOE_LAUNCH(0, 3)')
s=s.replace('MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1)', 'MOE_LAUNCH(1, 1); else MOE_LAUNCH(0, 1)')
(p/'dev/moe_w4a16_ss.cu').write_text(s)
# Embed in an independently named complete v3 candidate; remove unused resident banks.
v=(p/'cand_v3.py').read_text()
v=re.sub(r'^CUDA_SRC = .*$',lambda m:'CUDA_SRC = '+repr(s),v,flags=re.M)
v=v.replace('name="muse_prefill_grouped_w4a16"', 'name="muse_prefill_grouped_w4a16_ss"')
i=v.index('    try:\n        st["bank"] = _marlin_bank(weights)');j=v.index('    torch.cuda.synchronize()',i)
v=v[:i]+'    st["bank"] = None\n'+v[j:]
v=v.replace("    st['sms'] =", "    for key in ('w13', 'w13_sf', 'w2', 'w2_sf'):\n        st.pop(key, None)\n    st['sms'] =")
v=v.replace('Muse Spark Open prefill layer: v1.', 'Muse Spark Open prefill layer: v15, shared/shared WGMMA experts.')
out=p/'cand_v15.py'
if out.exists(): raise SystemExit('Refusing to overwrite '+str(out))
out.write_text(v)
print(out)
