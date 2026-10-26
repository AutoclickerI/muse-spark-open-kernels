# Public MoE mechanisms and the actual S19 contract

Sources: [operatorstack pin 6d9b7765](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_moe_w4a16) (MIT; SGLang-derived weighted sum Apache-2.0), [Lee v43d at 601c6541](https://github.com/lee101/openkernels/blob/601c65413496cf63f942b4f15ca087839ef2f5cc/muse_moe_w4a16/submission_v43d.py) (Apache-2.0). Full attribution is in `NOTICES.txt`. `review.json` and `sass_summary.json` identify the reviewed code and actual sm90a builds. SASS counts are static witnesses across template instantiations, not dynamic counts or causal attribution.

All ratios below are time port/S19, mean of eight paired seed ratios with 95% t7 intervals, retaining all four rounds. `550f0aa88785489f` is the native port cohort; `17b91ef9e95841e7` is the independently seeded matched-ABI/hybrid cohort. Different weight/input seeds mean their absolute times must not be compared across cohorts.

## M128 operator W13: a measured point, unresolved after adaptation

Native W13 is 0.952213 [0.894279,1.010148]. After writing S19's sorted BF16 intermediate and initializing its 64-part SS scratch, it is .981412 [.904507,1.058317], 133.648 versus 137.207 us. The interval spans one. The W13-only hybrid loses as a whole: 1.092380 [1.068740,1.116020].

`operator.py.inc::_decode_tile`, `_DECODE_ASM16`, `_w13_item` load packed NVFP4 words directly into the register layout of the MMA A operand. Bit permutation in prepare is lossless; masks/rotates place codes in BF16 bit positions, BF16x2 multiplication applies the block scale, and global scale is applied to FP32 accumulators. Only activations stage through XOR-swizzled shared memory. The item chooses 8/16/32 token rows, reusing a decoded weight tile across those rows. S19 instead has its original three expert branches, including CHUNK4 heavy WGMMA and shared packed-weight rings at lower counts. This source difference is established; a dequant-only speedup or causation is not measured by this experiment.

`operator_abi.py.inc` modifies only the four large-M GEMM helpers. `_abi_work` / `_abi_sorted` consume S19's route metadata, build work lists and map canonical pairs to S19 padded sorted rows; their cost is included. Matched route costs 8.665 versus 3.420 us, ratio 2.534060 [2.490510,2.577611]. The metadata overhead and loss of the original branch overlap prevent transferring the small point gain. No P4 patch is nominated.

## M128 operator W2: native boundary advantage does not survive S19's work

Native W2 is 0.911612 [0.874737,0.948486], but it emits BF16 raw expert Y. Its separate finalizer multiplies norm weights, computes square sums, normalizes and weights routes. S19 W2 already emits **FP32 norm-weighted Y and 64 square-sum tiles per row**. Comparing those W2 kernels alone omits work and halves the port's Y bytes.

`operator_abi.py.inc::_abi_w2_item` reads S19's sorted BF16 ACT through the row map, applies `(1 + post_norm_w)` to FP32 accumulators, stores FP32 Y and reduces each 64-channel tile into the correct SS slot. The complete ABI path passes TASK. Matched W2 is 1.025524 [.943486,1.107562], 80.726 versus 79.122 us; the W2-only hybrid loses 1.136557 [1.118357,1.154756]. The original apparent gain is not a borrowable mechanism.

## M128 Lee finalize: BF16/scalar-SS advantage is incompatible with S19 buffers

Native X2 finalize is 0.914463 [0.870148,0.958778]. `lee_x.cu.inc::finish_x` retains the donor's vector loop: each warp handles a 256-channel slice, each lane accumulates eight channels across all eight routes, reading BF16 Y and one FP32 square sum per pair. It needs no CTA-wide alpha publication/barrier.

S19 supplies FP32 Y and 64-part SS. `abi_finalize.cu.inc` preserves the vector loop but reads FP32 Y and performs a warp reduction of all 64 SS parts. It is 1.350141 [1.307908,1.392374], 7.576 versus 5.616 us. Its complete S19 hybrid loses 1.014787 [1.009616,1.019959]. A faster native finalizer on different buffers does not establish a transferable win. The adaptation is explicitly marked as modified Lee code and carries the full Apache text in the payload.

## Other M and Lee expert-wide RS-WGMMA

All whole ports lose at M1/8/16/32/128/256; full ratios and CIs are in `H200.md`. M1–32 S19 fuses W13, W2 and finish, so separate S19 phase ratios are unavailable. Native donor phase times are retained for P3a in `P4_HANDOFF.json` and `receipt_data.json`; inventing separate fused phase costs would require another experiment.

Lee's X source dequantizes a weight tile once per expert/token pass, uses gathered SW128 activation rows and an RS-WGMMA producer/consumer ring. Both published X configurations were tested: RPW2/NCW8 and RPW1/NCW16, TPX64. Actual sm90a SASS has HGMMA (768/384 across generated X1/X2 instantiations); the two original small-M F kernels have HMMA and no HGMMA. X's gather plus per-CTA route-list construction and its larger resident work units lose on this cohort. No dequant-only ablation isolates which term causes the whole-port loss.

At M256, matched O W13 is 1.016750 [.964685,1.068815], O W2 1.093285 [1.032576,1.153994], and Lee finalize 1.541246 [1.431000,1.651492]. O132's finish ratios .994688/.986545 at M128/256 time the **unchanged S19 finalizer** and are controls, not donor mechanisms. None establishes a new winning implementation. The tested port recipe is closed; the MoE structural ceiling remains open.
