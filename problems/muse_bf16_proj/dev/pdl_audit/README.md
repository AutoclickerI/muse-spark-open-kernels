# S14 PDL load audit (P7)

Frozen base: `problems/muse_decode_layer/cand_r3_best7_s14_final.py`, SHA256
`f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.
This is glue/load ordering work; projection algorithms, layouts, split counts,
attention algorithms and expert consumers remain S14's choices. P10b owns projection algorithms.

Current handoff: `patch_metadata_s14.py` (`f39bc1f3`) produces the exact measured
metadata-only candidate `5f4e6aa9`. Its small paired gain is **provisional**:
unchanged JIT kernels were compiled separately in the original lab namespaces.
Shared-device-function confirmation is ready but quota-blocked until 09:00 KST.
No incumbent change or promotion was made; no lane-owned lab remains pending.

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

`patch_metadata_s14.py::patch(source)` is the self-contained integration artifact
and rejects any base hash except S14 final. `patch.py` builds the rejected broad
register-hoisted experiment. Run `python3 problems/muse_bf16_proj/dev/pdl_audit/patch.py` to write
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

H200 job `8db4213240c6420d` completed on the attention import-time host
(payload `c638acca`, candidate `04d4ba68`, budget 550 s). The only
authorized lab host for this work is attention;
decode checks remain reserved for integration and P10b.


Additional native admission: `check_glue.py` passes 24 cells / 384 changed-input
replays bit-exactly; `check_partial.py` passes eight cells / 192 changed-input
replays bit-exactly (L1/L8, both windows, BF16 and seven FP32 splits, changing
positions and non-unit scales). The latter executes the actual partial-attention
kernels on sm120, including cluster support, rather than a launch stand-in.

The standalone prior full-decode recipe is rebased in `patch_d1_s14.py`
(`5716052b`), producing `d1_s14_trial.py` (`54515aa9`). Its S13 evidence does not
establish a new S14 gain. The broad patch tests its glue change as part of all fixes.

`build_hint.py` creates a low-register follow-up: preserve the original GEMV
pipeline and request its first tile's cachelines before wait. This retains all
other fixes while removing the long first-weight register lifetime. It is not
part of job `8db4213240c6420d`; its provisional H200 result is arm B in `0f74021a2c53445e`.


Resource attribution: native MoE flag95 remains 119 registers, VQ8 main ST6
remains 161 registers, and VQ8 combine remains 64 registers. Partial L1 rises
120 to 126 registers; L8 remains 254. Native shared/local allocations are
unchanged. In the low-register follow-up, representative balanced/GI kernels
return from 150 to 48 registers while retaining one pre-wait cache request
instruction. These compiler facts do not establish H200 performance.


## H200 8db4213240c6420d — all-fixes loss

All twelve fixtures / sixteen seeds passed; import-time lab took 192.2 s.
Ratios below aggregate each seed's 46 sliding + 16 full layer times; errors
are standard errors across the sixteen seeds.

| Load | A:A | All fixes / S14 |
|---|---:|---:|
| decode1 | .998631 ± .001151 | 1.108637 ± .002651 |
| decode16 | 1.002417 ± .000638 | 1.242469 ± .002222 |
| decode32 | .998995 ± .000630 | 1.231254 ± .002164 |
| verify1 | .990772 ± .000825 | 1.223309 ± .002500 |
| verify16 | .997619 ± .000587 | .997104 ± .000547 |
| verify32 | .999442 ± .000450 | 1.005398 ± .000460 |

Aggregate six-load speedup .884950; per-seed speedup .885126 ± .000701.
A:A per-seed speedup 1.002036 ± .000348. No all-fixes integration patch wins.
The strong losses concentrate at the loads using small GEMVs. Their changed first-weight tiles raised registers and altered the async
pipeline. The exact F/M attribution below isolates those three projection
prologues; native and other metadata fixes are identical in F and M.

`build_followup.py` prepares a same-run attribution: F repeats the exact losing
recipe, B uses low-register weight hints, M keeps only metadata ordering changes,
and D applies only the rebased prior d1-full glue recipe. A/A2 are unchanged S14.
All share baseline choices and weights; repaired native modules are shared only
when their source bytes match. The new input seeds are held out. Per-arm output
quality also records bit-exactness, route differences and worst L2 without hiding
rows that change routes. Its completed H200 result is below.


`weight_pipeline_mechanism.json` narrows the mechanism further. The new
conditional first tile changes the software pipeline as well as register
lifetime: balanced GEMV LDGSTS instruction count 40→8 and LDG count 0→16,
while tensor-core instruction count remains eight. The hint variant restores
40 LDGSTS / zero LDG and 48 registers. Fixed-time losses in the first run are
similar between sliding/full at each small-M load: d1 +8.25/+7.96 us,
d16 +36.24/+38.64 us, d32 +44.46/+42.98 us, v1 +26.58/+26.70 us.
These are compiler evidence and a whole-layer signature; the direct attribution
is in completed job `0f74021a2c53445e` (payload `f9f651bb`).

Every follow-up variant passed normal and forced full-layer screens: F/B/M
13 tests, 12 graph shapes, 24 static checks; D has 25 static checks including
its extra native glue source. Local six-arm orchestration passed all twelve
fixtures / sixteen seeds, maximum whole-output L2 .001338. Its complete output
is 8,027 bytes, below the 9,000-byte single-payload cap. Native admission remains
bit-exact as recorded above. No Hopper performance is inferred from local timing.


Pairing audit before integration: original lab namespaces retained separate
unchanged Triton JIT objects. Shifted source lines changed cache keys even when
executable source matched. Therefore the small 0f74021a gains remain provisional
pending shared-object confirmation. `build_followup.py --shared-jit` shares
source-identical JIT objects only when their complete JIT dependency closure is
also unchanged. Changed kernels remain private; Python stage functions keep
their own globals. Exact candidate source hashes and baseline plans are retained.


## H200 0f74021a2c53445e — attribution, provisional small gain

All twelve fixtures / sixteen new seeds passed; 223.4 s import-time lab.
Zero changed route memberships; worst whole-output L2 .000289 across all arms,
including A:A. Ratios below are arm/A latency, aggregated per seed.

| Load | A:A | F register tile | B cache hints | M metadata only | D prior d1-full |
|---|---:|---:|---:|---:|---:|
| decode1 | 1.002776 | 1.098952 | .987959 | .990014 | .991264 |
| decode16 | .996629 | 1.237942 | .993301 | .996089 | .996302 |
| decode32 | .999669 | 1.228076 | .998670 | .997095 | .998514 |
| verify1 | .997464 | 1.223450 | .998656 | .994498 | .997399 |
| verify16 | 1.000032 | .998923 | .999736 | .998415 | .999439 |
| verify32 | .999080 | 1.004844 | 1.005680 | 1.005023 | 1.000135 |

Aggregate six-load speedups: A:A 1.0007067; F .8870202; B 1.0026880;
M 1.0031687; D 1.0028444. M's per-seed speedup is 1.003170 ± .000317;
A:A 1.000730 ± .000180. M/A2 speedup is 1.002438 ± .000346. Metadata-only
retains all projection functions and source strings. Verify32 still regresses
~.5%. D's targeted full decode1 latency ratio is .981287 ± .002153; the larger
13%-of-wall glue target remains unmet.

The source/dependency-identical kernels had distinct compiled CUDA function
handles in the original namespaces, confirmed by `check_pairing.py`. Its
shared-object version proves actual device function equality while changed
kernels remain private. The revised local six-arm lab passes all twelve fixtures
and sixteen seeds, with 26 unchanged JIT objects shared for M and 35 for D;
its output is 7,308 bytes. This verification addresses small-gain attribution,
not the large F regression's compiler mechanism.

## Capacity block and resume

The confirmation payload `lab_pdl_shared_followup.py` (`6cb428bc`) uses sixteen
fresh seeds starting at 3100973. Its send was **not accepted**: all H200 hosts
were out of scored/check quota until 2026-10-26 09:00 KST. The lane's own
reset-wait sender (PID 3107418, wrapper `c8ab1506`) was cancelled; no third job ID
exists, and neither completed job remains pending. `handoff.json` records the
provisional status and exact patch/candidate hashes.

Rebuild the complete confirmation from committed sources, then resume after reset:

```sh
python3 problems/muse_bf16_proj/dev/pdl_audit/build_followup.py --shared-jit
python3 tools/h200_lab.py send problems/muse_bf16_proj/dev/pdl_audit/lab_pdl_shared_followup.py --host muse_attention_decode --import-budget 550 --priority 1 --note "P7 S14 preload shared-JIT confirmation; A:A and F/B/M/D,16 new seeds"
```

Decode quota remains reserved for integration/P10b. Integration may instead
confirm the exact metadata patch in its own paired composition, sharing
unchanged kernels. Do not promote solely on the provisional .3% gain.
