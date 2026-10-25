# S14 PDL load audit (P7)

Frozen base: `problems/muse_decode_layer/cand_r3_best7_s14_final.py`, SHA256
`f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.
This is glue/load ordering work; projection algorithms, layouts, split counts,
attention algorithms and expert consumers remain S14's choices. P10b owns
projection algorithms. Nothing here is promoted without the paired H200 result.

## Scope and safety

`audit_native.py` extracts every native source containing a PDL wait, release
or launch attribute, including composed `PARTIAL_FAST_CUDA` and instantiated
`CD_CUDA`. All ten sources compile at `-O3 -lineinfo` for sm_90a, covering 72
release/wait specializations (the two PARTIAL source names currently contain
identical bytes). `audit_triton.py` compiles all 15 Triton PDL kernels and
assembles their generated PTX for sm_90a. Original/fixed SASS and line tables
are in `native/{baseline,fixed}/`; these rebuildable dumps are ignored by git.
The JSON audits retain kernel names, waits, preload instructions, resources,
and source-line inventories. SASS PCs alone are not control-flow proofs:
notably, VQ8 producer warps never execute the consumer warps' wait.

Only immutable weights or timed-call input data move before synchronization.
The predecessor's QKV/activation/partial/route outputs remain dependent. A gate
or metadata buffer produced by an earlier kernel in the same PDL chain is not
assumed ready merely because it precedes the immediate producer. This follows
the [CUDA PDL synchronization contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).

Large weight/history streams cannot all remain in registers across a wait;
the patch preloads their bounded first tile, retaining the subsequent streaming
pipeline. Route-dependent expert weight/packed-scale addresses remain after
routing; the two fixed-size 32-expert global-scale vectors can be loaded early.

## Native inventory

| Source / kernels | Independent data after the wait in S14 | Change / reason retained |
|---|---|---|
| GLUE_CL_CUDA boundary_cl (14 variants), route_cl (5) | Boundary alpha/beta/prew/postw; route prew/bias/local_start. 17/19 have zero early loads; C2/256 boundary already preloads. | Inline PTX vector/scalar loads at intended source preload sites. This folds the `e021d263` decode1-full repair into the combined audit. |
| CUDA_SRC attn_combine (L1/L8) | v_scale | Load v_scale at entry. Partials and QKV gate stay dependent. |
| CUDA_SRC attn_main (L1/L8), ATTN_PORT_CUDA_SRC attn_main (L1/L8) | No predecessor wait; these launch normally and release their successors. | Dumped; no missing preload boundary. |
| PARTIAL_CUDA / PARTIAL_FAST_CUDA attn_main (L1/L8 each) | RoPE positions/inv_freq and k_scale/v_scale. Cache-location loads are intended before wait; first page/history stages already issue cp.async before wait. | Move angle-input/scale loads before wait, pin input metadata/page loads with PTX, reuse scale values later. Query/new KV remain after wait. |
| VQ8_CUDA vq8_main (ST2/4/6/8) | Consumer inv_freq and cache locations. Consumer k/v scales already precede wait. | Preload inv_freq and first portion's cache locations; retain later-portion location streaming. KV-history producer already runs independently without the consumer wait. |
| VQ8_CUDA vq8_combine | v_scale | PTX load before wait. Scheduling prefix, partials and QKV gate stay dependent. |
| CD_CUDA coded_gemv (NT1/RT1/PIPE2/V3) | Later compressed weight tiles and data-dependent escape values | Initial scales/offsets/weight pipeline already uses inline PTX before wait with prew=1. Retain streaming; no repair. The prew=0 control intentionally waits first. |
| GF_CUDA direct (28 specializations, including trace controls) | Subsequent weight-pipeline tiles | Initial DEPTH weight tiles already precede wait using vector PTX loads. Retain. |
| SMALLM_VAR_CUDA moe_var (3 PDL flag cells) | gs13[e]/gs2[e]; expert-indexed packed weights/scales and post-norm rows | Load all 32 global scales per warp before wait; select via warp shuffle after routing. Expert-dependent addresses remain in their routed pipeline. |
| DENSE_CHAIN_CUDA dense_island (6) | Weight streaming and later boundary parameters | Optional bulk weight prefetch already precedes wait. All S14 dense_chain rows, including other, are false; this family is not dispatched by the base. No change. |

## Triton inventory

| Kernels | Finding and repair |
|---|---|
| _gemv_kernel, _gb_kernel, _gi_kernel | _gemv has L2 hints but no first tile in registers; balanced/GI waits first. Pin one vectorized first weight tile before wait; retain original later-tile arithmetic/order. Registers rise (representative _gb/_gi 48 to 150), so overlap can lose to occupancy. |
| _gate_split_kernel | Whole independent gate GEMV already precedes the last-owner wait. No change. |
| _qk_prep_kernel | Positions/frequencies/cos/sin already preload; pin scalar/input loads. |
| _best7_old_qk_prep_kernel | All metadata/scales/RoPE inputs follow wait. Move/pin them before it. |
| _boundary_fast, _route_fast, _best7_old_boundary_kernel, _best7_old_route_norm_kernel | Main weight vectors already precede wait in this compiler. Pin their loads without losing vectorization; dependent branches/partials remain after. |
| _boundary_late | All weights intentionally follow wait. Preload alpha/beta/prew/postw; all other math retained. |
| _attn_split_kernel | Sequence length precedes wait; k_scale/page/history loads follow it. Preload scale, first page rows and strictly historical KV rows. The mixed tile's new-token rows load after wait; later history stays streamed. |
| _attn_combine_kernel | Preload v_scale; partials/gate stay after wait. |
| _attn_sumcast_kernel, _norm_gate_kernel | Only predecessor/chain-produced buffers; no safe immutable preload to move. |

## Build and validation

`patch.py::patch(source)` is self-contained and rejects any base hash except
S14 final. Run `python3 problems/muse_bf16_proj/dev/pdl_audit/patch.py` to write
the candidate, then run the two audit scripts with `--candidate ... --label fixed`.
`report_sass.py` maps native load instructions back to source lines. The normal
and forced full-layer screens use `../verify_epilogue/screen.py`; unchanged
native sources reuse source/compiler/flag-matched compile receipts while every
changed source compiles again. Failed local builds remain local.

`build_lab.py` embeds the exact baseline and candidate plus public synthetic
weight/input generators. The import-time attention host supplies no weights.
One baseline prepare chooses algorithms and banks, then both arms share those
choices. Changed native module objects/caches remain private. The lab compares
A/A2/B over all twelve fixtures, sixteen seeds and twelve rotated cold READ
rounds, with per-seed gates and 46/16 weighted load ratios. It preserves the
complete compressed samples beneath the wrapper's log cap. `check_lab.py`
tests this orchestration locally using S14's portable paths; it does not claim
to execute Hopper-only VQ8 instructions on sm120.

H200 job `8db4213240c6420d` is pending on the attention import-time host
(payload `c638acca`, candidate `04d4ba68`, budget 550 s). The only
authorized lab host for this work is attention;
decode checks remain reserved for integration and P10b.
