from pathlib import Path
import re
p=Path('problems/muse_prefill_layer')
c=(p/'dev/moe_w4a16.cu').read_text()
c=c.replace('constexpr int BN = 128;', 'constexpr int BN = 256;')
c=c.replace('constexpr int W_BYTES = 4 * 2048;', 'constexpr int W_BYTES = 2 * 2048;').replace('constexpr int SF_BYTES = 4 * 256;', 'constexpr int SF_BYTES = 2 * 256;')
c=c.replace('constexpr int STAGE_BYTES = X_BYTES + W_BYTES + SF_BYTES;', 'constexpr int COPY_BYTES = X_BYTES + (MOE_WGMMA ? W_BYTES + SF_BYTES : 0);\nconstexpr int STAGE_BYTES = ((COPY_BYTES + 1023) / 1024) * 1024;')
c=c.replace('return 2 * (MODE == 1 ? 1 : 2) * X_BYTES;', 'return 2 * X_BYTES;')
c=c.replace('float (&acc)[64]', 'float (&acc)[128]').replace('float (&d)[64]', 'float (&d)[128]').replace('i < 64; ++i', 'i < 128; ++i')
a=c.index('  asm volatile(\n',c.index('uint64_t desc = make_desc_b'));b=c.index('\n#else',a)
reg=', '.join('%'+str(i) for i in range(128))
outs=',\n        '.join(', '.join('"+f"(d[%d])'%i for i in range(j,j+8)) for j in range(0,128,8))
c=c[:a]+f'''  asm volatile(
      "{{\\n.reg .pred p;\\nsetp.ne.b32 p, %133, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{{{reg}}}, {{%128, %129, %130, %131}}, %132, p, 1, 1, 0;\\n}}\\n"
      : {outs}
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d));'''+c[b:]
c=c.replace('((size_t)(e * p.nfb + fb) * KT) * W_BYTES','((size_t)(e * (p.nfb / 2) + fb / 2) * KT) * 8192 + (fb % 2) * 4096')
c=c.replace('((size_t)(e * p.nfb + fb) * KT) * SF_BYTES','((size_t)(e * (p.nfb / 2) + fb / 2) * KT) * 1024 + (fb % 2) * 512')
c=c.replace('mbar_expect_tx(&full[s], STAGE_BYTES);','mbar_expect_tx(&full[s], COPY_BYTES);')
c=c.replace('        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * W_BYTES, W_BYTES, &full[s]);\n        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * SF_BYTES, SF_BYTES, &full[s]);', '''#if MOE_WGMMA
        bulk_g2s(st + X_BYTES, wsrc + (size_t)k * 8192, W_BYTES, &full[s]);
        bulk_g2s(st + X_BYTES + W_BYTES, ssrc + (size_t)k * 1024, SF_BYTES, &full[s]);
#endif''')
c=c.replace('float acc[2][64];','float acc[128];')
a=c.index('    uint32_t A0[2][4]');b=c.index('    // prologue:',a)
c=c[:a]+'''    uint32_t A0[4], A1[4];
    uint4 raw;
    uint2 sfw;
    auto load_words = [&](int st_idx, int k) {
#if MOE_WGMMA
      const uint8_t* st = smem + st_idx * STAGE_BYTES;
      raw = *reinterpret_cast<const uint4*>(st + X_BYTES + cw * 2048 + t * 16);
      sfw = *reinterpret_cast<const uint2*>(st + X_BYTES + W_BYTES + cw * 256 + slot * 8);
#else
      size_t wp = ((size_t)(e * (p.nfb/2) + fb/2) * KT + k) * 8192 + (fb%2)*4096;
      size_t sp = ((size_t)(e * (p.nfb/2) + fb/2) * KT + k) * 1024 + (fb%2)*512;
      raw = *reinterpret_cast<const uint4*>(p.wblob + wp + cw * 2048 + t * 16);
      sfw = *reinterpret_cast<const uint2*>(p.sfblob + sp + cw * 256 + slot * 8);
#endif
    };
    auto issue = [&](uint32_t (&F)[4], const uint8_t* xs, int b, int scale_d) {
      fence_acc(acc);
      wg_fence();
      mma_block(acc, F, xs, b, scale_d);
      wg_commit();
      fence_acc(acc);
    };
'''+c[b:]
c=c.replace('load_words(s);','load_words(s, 0);',1)
c=c.replace('load_words(s);','load_words(s, k + 1);')
c=re.sub(r'#pragma unroll\n\s*for \(int j = 0; j < 2; \+\+j\) make_frag\(raw\[j\]\.([xyzw]), e4m3x2_to_bf16x2\(sfw\[j\]\.([xy]) (.*?)\), lo_a, hi_a, (A[01])\[j\]\);',r'make_frag(raw.\1, e4m3x2_to_bf16x2(sfw.\2 \3), lo_a, hi_a, \4);',c)
c=c.replace('keep(A1[0]); keep(A1[1]);','keep(A1);').replace('keep(A0[0]); keep(A0[1]);','keep(A0);')
c=c.replace('    fence_acc(acc[0]);\n    fence_acc(acc[1]);','    fence_acc(acc);')
a=c.index('    // ---- epilogue:');b=c.index('\n  }\n  if (t == 0)',a)
c=c[:a]+'''    // Two consumer groups hold matching gate/up rows for MODE1, or consecutive
    // down-projection features for MODE0. BF16 shared epilogue exchanges gate/up.
    const float gsc = p.gs[e];
    if (t == 0) bulk_wait_read();
    named_bar(3, 256);
    const int r0 = warp * 16 + (lane >> 2);
    const int c2 = 2 * (lane & 3);
#pragma unroll
    for (int i = 0; i < BN / 8; ++i) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int rho = r0 + 8 * h;
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int n = 8 * i + c2 + q;
          auto bv = __float2bfloat16_rn(acc[4*i+2*h+q] * gsc);
          *reinterpret_cast<__nv_bfloat16*>(my_epi + n*128 + (((rho>>3)^(n&7))<<4) + (rho&7)*2) = bv;
        }
      }
    }
    named_bar(3, 256);
    if constexpr (MODE == 1) {
      if (cw == 0) {
        auto gate = reinterpret_cast<__nv_bfloat16*>(epi);
        auto up = reinterpret_cast<__nv_bfloat16*>(epi + X_BYTES);
#pragma unroll
        for (int i=0; i<X_BYTES/2/128; ++i) {
          int idx=t+i*128;
          float gg=__bfloat162float(gate[idx]);
          float uu=__bfloat162float(up[idx]);
          gate[idx]=__float2bfloat16_rn(gg/(1.f+__expf(-gg))*uu);
        }
      }
    }
    fence_async_smem();
    named_bar(3, 256);
    if (t == 0 && (MODE == 0 || cw == 0)) {
      const int kto = MODE == 1 ? fb : fb*2+cw;
      bulk_s2g(p.y + ((size_t)tt * p.kt_out + kto)*X_BYTES, my_epi, X_BYTES);
    }
'''+c[b:]
c=c.replace('MOE_LAUNCH(1, 6); else MOE_LAUNCH(0, 5);','MOE_LAUNCH(1, 4); else MOE_LAUNCH(0, 4);').replace('MOE_LAUNCH(1, 2); else MOE_LAUNCH(0, 1);','MOE_LAUNCH(1, 1); else MOE_LAUNCH(0, 1);')
assert 'raw[j]' not in c and 'acc[0]' not in c
(p/'dev/moe_w4a16_bn256.cu').write_text(c)
h=(p/'dev/moe_host.py').read_text()
a=h.index('def load_module(');b=h.index('\n\n@triton.jit',a)
loader=h[a:b].replace('def load_module(', 'def load_module256(').replace('_mod','_mod256').replace('load_mod256ule256','load_module256').replace('src = open(src_path).read()','src = CUDA_SRC256').replace('name="muse_prefill_grouped_w4a16"','name="muse_prefill_grouped_w4a16_bn256"')
fn=h[h.index('def experts_tiled('):].replace('def experts_tiled(', 'def experts_tiled256(').replace("st['tiled_mod']", "st['tiled_mod256']")
fn=re.sub(r'\bBN\b','256',fn).replace('32, 64, 64, 1','64, 64, 64, 1').replace('16, 64, 64, 0','32, 64, 64, 0')
s=(p/'cand_v3.py').read_text()
a=s.index('    if T >= BIG_TOKENS or st["bank"] is None:');b=s.index('    if T <= 8:',a)
s=s[:a]+'''    if T < 8192:
        moe = experts_tiled(pooled, topk_w, slot_e, slot_pos, counts, st)
    else:
        moe = experts_tiled256(pooled, topk_w, slot_e, slot_pos, counts, st)
'''+s[b:]
a=s.index('    try:\n        st["bank"] = _marlin_bank(weights)');b=s.index('    torch.cuda.synchronize()',a)
s=s[:a]+'    st["bank"] = None\n'+s[b:]
s=s.replace("st['tiled_mod'] = load_module(None)", "st['tiled_mod'] = load_module(None)\n    st['tiled_mod256'] = load_module256(None)")
s+='\n\nCUDA_SRC256 = '+repr(c)+'\n\n_mod256 = None\n\n'+loader+'\n\n'+fn
(p/'cand_v12.py').write_text(s)
