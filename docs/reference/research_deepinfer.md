Read when adapting DeepInfer's dependency publication, operand pipelines, routing selection or replay state.
Serves: G2.

# DeepInfer designs that transfer to Muse decode

Source audit, not an H200 benchmark. DeepInfer `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`
targets SM100. Its DeepGEMM gitlink `1fc9c66975ee322d628b3288534ca0e9ce12a64d` is the
implementation basis; inspected mainline `5081f8513af57e61696de4f4f04b0e707ffb503e`
adds Mega-mHC's replay epoch fix. FlashMLA gitlink is `2e5429fc5653bab6e081f09477126f731882a6a9`.
Reported performance belongs to SOTA; Muse measurements/closed recipes belong to FINDINGS;
proposed experiments belong to PLAN. No aggregate source speedup predicts a Muse gain.

## Mega-MoE dependency and operand pipelines

`MLP.mega_moe_forward` consumes indices/weights already produced by `gate_forward`.
Mega-MoE buckets and dispatches routes; it does not calculate logits/select top-k. [call site][di-mlp]

Separate activation+scale and weight+scale TMA warps feed a joint full barrier. Activation loading
waits for dispatch/W13 readiness, while weight loading waits only for its operand slot to become
free. Pipeline indices continue across tasks rather than drain/reinitialize per output tile.
This retains operands, unlike an L2 hint that still needs a subsequent register/shared load.
The source also double-buffers combine loads and fetches metadata ahead of stores. [kernel][moe-kernel]

Gate/up rows interleave by eight in weights and scales. W13 applies SwiGLU and writes the W2 input
into a global ring; fusion does not keep that entire intermediate in registers. After TMA stores
complete, each W13 N-block toggles a bit with `red_xor_rel`. `L2KBlockDependency` maps each W2
K-block to the precise W13 bits it needs. Acquire polling caches observed pending masks; generation
parity and bounded ring capacity distinguish reused slots. This permits partial readiness rather
than an expert-wide completion wait. Publishing every tiny Muse tile individually could erase
the benefit; the notification granularity and consuming K-slice must agree. [scheduler][moe-sched]

The scheduler warms enough W13 waves before admitting W2 so waiting consumers cannot occupy all
producer capacity. It checks required W13 tasks were fetched from every locality queue; helpers
bound live pool blocks and ring capacity. Do not substitute an arbitrary count of completed
producers for the named subset a consumer needs. [warmup and next-task logic][moe-sched]

Released FP8×MXFP4 uses Blackwell block-scaled MMA; BF16 Mega-MoE uses BF16 weights too.
Neither is a Hopper BF16-activation/NVFP4 adapter. Unmerged SM90 PR #454 at
`d5925deb53a26895efad49395831f0bcf30d13b1` decodes into FP8 shared operands and uses FP8
activations, so it is not admissible unchanged. Its 0.5 permutation/sign PTX operations per
weight excludes selector/address/scale/store work and is not comparable to a full SASS decoder
count. E2M1 1.5 × E4M3 1.125 = 1.6875 fits BF16, not E4M3; divide/cast/restore cannot recover
lost significand bits. [proposal decoder][sm90-proposal] · [LUT][sm90-lut]

DeepGEMM puts route weights in W13 before W2. Muse's per-expert post-norm prevents copying
that placement (semantic contract: [research_moe_decode](research_moe_decode.md#model-invariants)).
Communication elimination supplies no gain to the one-GPU problem.

## Mega-Gate selection and glue ownership

Mega-Gate keeps split score scratch in global memory and releases token blocks through
`ScoreBarrier`; one launch does not eliminate the score-gather join. Its local selection sorts
each lane's groups of four once with a five-comparison network, retains permutation/cursor
state, and offers only each group's next candidate to warp max/min. Smallest ID wins ties;
only the winner advances, and an output lane owns each selected ID. This avoids repeated local
scans and dynamic winner arrays. [gate kernel][gate-kernel] · [selection epilogue][gate-epi]

Its scoring is sqrt-softplus and weights BF16; Muse needs sigmoid and FP32 hi+lo accumulation.
Import selection mechanics, not that arithmetic. Bias changes selection, not unbiased mixture
weight normalization. A separate router may also lose Muse's pool/router input sharing.

Mega-mHC carries coefficients/vector values across row reductions, loads next coefficients while
stores drain, and recycles retired producer roles into normalization. Its shifted-coefficient
mode changes architecture and cannot transfer. Keep Muse FP32 residuals and BF16 materialization
points. TileKernels' last-split epilogues and producer-native output layouts are useful smaller
patterns when a whole megakernel loses parallelism. [mHC roles][mhc-kernel] · [epilogue][mhc-epi]

**Replay freshness:** Mega-mHC mainline replaces grid-ID tags with `Epoch` because a captured
graph node keeps its grid ID across replays. The last retired CTA advances the workspace epoch.
This proves `%gridid` alone is insufficient; it does not establish a bug in every other tagged
protocol, which may reset/order differently. [epoch fix][mhc-epoch]

## Attention pipeline primitives

FlashMLA's fused sparse d512 SM100f kernel does not fit Muse's all-key d64/page1 H200 call.
Its transferable details are independent QK/PV readiness, bounded issue distance protecting two
probability buffers, page-index lookahead, compact hot loops and producer-native layouts.
The adaptive issue loop checks whether previous PV can run before waiting on the next QK;
`num_qk_issued <= num_sv_issued+1` protects buffer ownership. Gather coordinate generation,
raw transport and dequant have separate readiness. [core][fm-core]

Q norm's scalar denominator is applied after QK while QK overlaps norm-statistic generation.
Muse's learned elementwise factors and BF16 rounding mean algebra alone is no correctness proof.
Constructing `uint4` from independent conversion results can avoid register-quad alignment MOVs;
inspect SM90 SASS before claiming savings. Blackwell TMEM, gather4 and CLC do not directly port.

## Tiny-router selection versus DeepSelect

DeepSelect's long-row filtering/radix work is not a drop-in top-8-of-256 replacement. Durable
primitives are ballot/popcount prefixes, `ffs` enumeration of candidate bit masks, one winner
per output lane and loading the first candidate before count exchange. Its denormal-FP32
integer-increment trick requires exact bounded integers and no FTZ; it is not numerical score
arithmetic. Preserve FP32 router scores. [filter/move implementation][ds-core]

[di-mlp]: https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L359-L563
[moe-kernel]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh#L57-L1528
[moe-sched]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh#L16-L475
[sm90-proposal]: https://github.com/lengrongfu/DeepGEMM/blob/d5925deb53a26895efad49395831f0bcf30d13b1/deep_gemm/include/deep_gemm/impls/sm90_nvfp4_mega_moe_fused.cuh#L34-L278
[sm90-lut]: https://github.com/lengrongfu/DeepGEMM/blob/d5925deb53a26895efad49395831f0bcf30d13b1/deep_gemm/include/deep_gemm/quantization/nvfp4_dequant.cuh#L18-L23
[gate-kernel]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh#L25-L447
[gate-epi]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh#L19-L215
[mhc-kernel]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh#L35-L696
[mhc-epi]: https://github.com/deepseek-ai/DeepGEMM/blob/1fc9c66975ee322d628b3288534ca0e9ce12a64d/deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh#L139-L428
[mhc-epoch]: https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh#L11-L80
[fm-core]: https://github.com/deepseek-ai/FlashMLA/blob/2e5429fc5653bab6e081f09477126f731882a6a9/csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.cuh#L183-L1506
[ds-core]: https://github.com/deepseek-ai/DeepSelect/blob/bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18/csrc/cuda_kernels/common_parts.cuh#L940-L1480
