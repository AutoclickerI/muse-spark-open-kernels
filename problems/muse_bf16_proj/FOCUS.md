# Projections and glue in the decode layer (P7)
Serves: G1, G2 (dense GEMVs, verify dense GEMMs, glue). Levers: [NEXT_CHECKS](../muse_decode_layer/dev/NEXT_CHECKS.md).

Incumbents in S7 (`cand_r3_best7_s7.py` bd3932d1cc3c0c82d60145772148fd19fc779465dd360bda610d214385401390):
- Projections: best7 per-role dispatch (cuBLAS/Lt, custom GEMV) + O config6 at decode 1, `dev/gemv_fixed/projection.py`
  c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe.
- Glue: S6 Triton `_boundary_fast`/`_route_fast` at best7 geometry; attn_main rows sum+cast qkv in torch (4.7-5.5 us).

Ready, not in S7: `../muse_decode_layer/cand_r3_best7_s7_p7c.py`
fbe84edc697fa13402bde0ab66dafe8e48af6a7c28e1a10069588316dac93d8c = S7 + attn early release (`dev/attn_pdl`, `4d995f13`:
decode 1 -0.95/-2.06 us) + M=128 qkv split-K (`dev/verify_splitk`, `b15a3bde`: r 0.8777) + cluster glue at t<=32
(`dev/glue_cluster`, `1365a290`: -1.9 to -3.2 us per layer). Paired chain labs, cold L2; whole layer unmeasured; 13/13 local.
Staged: `cand_r3_best7_s7_p7d.py` 3426c79773b1d03152ff39e26761167b0b0a89ce06f212242c68f03bd90fe075 = p7c + BF16 qkv
front (`dev/attn_front/patch_front_bf16.py --fix --layer`); its flags follow `8665cfc9`; 13/13 local.

In flight (proj host): send 8665cfc9216f16bc attn_front v2 (fused PDL sum+cast or GEMV split-K fixup -> BF16, x PDL
attn_main; layer-ext arms at decode 16 sliding; local bit-exact, -2 to -7 us); send 0169172e2a37429b gemv_roof (S6 GEMV,
cuBLAS, CUDA-core GEMV vs a plain read roof at M=1/16/32).
Measured losses (exact recipes; FINDINGS): attn_main reading FP32 partials `aab93d73`/`6ad1f245` (+3 to +6 us); FA3
schedule on a side stream `e1a56ee7`; verify split-K beyond M=128 qkv `b15a3bde`; cluster glue at t=128/256 `1365a290`;
bulk ring `87b2667f` (write flush); cooperative glue `9eeb39ba`; fused glue `80b70f7f` (bug); register-MMA GEMV
`95ab383d`; persistent chain `87dc944b`; side-stream L2 prefetch `d5a90258`; hardware compression `4233660b`.
Next: read `8665cfc9`, set p7d's flags, hand off; read `0169172e`, add a CUDA-core GEMV variant where M=1 trails the
read roof; `route_cl` emits MoERouteMeta v1 (711 words equal, local) for the small-M MoE consumer.
