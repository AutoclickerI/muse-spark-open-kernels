# Persistent grouped TMA experiment

Closed H200 experiment; withdrawn and parked at `dev/parked/cand_tma_grouped.py`.
SHA256 **b1209b2ac260654999c11a915606dcfb784dc3373c6e3a26da001a3b2c9e163c**.
The complete `cand_v9_phase.py` source is its unchanged prefix; original prepare and
forward remain the incumbent. No repository imports, judge launch or commit by this lane.
This is separate from the MoE lane's V19 scheduling portfolio.

One persistent kernel builds stable expert/token lists, sorts experts by load, creates a
tile-prefix schedule and runs W13+SwiGLU, W2, per-expert output RMSNorm and weighted sum.
Hot experts use32/64-token tiles split across CTAs; small experts use swap-AB m64n8/16
WGMMA with packed weights decoded into registers. A raw17-derived TMA ring stages two
m64 weight/scale blocks; indexed BF16 activation gathers use cp.async and128B swizzles.
No activation/weight precision is reduced. W13 writes BF16 SwiGLU, W2 accumulates FP32.
Two resident-grid barriers join the GEMMs and final reduction; no separate routing,
SwiGLU, norm/add kernels or norm-partial atomics. Grid capacity is checked before capture.
This does not implement DSMEM multicast or require a cluster launch.

Choices `(max_token_tile, ring_stages, SM_grid_fraction, minimum_token_tile)` are
32/64,3/5,1 or.5,8/16 (12 combinations). Four SM90a native specializations compile and
contain m64n8/16/32/64 WGMMA and bulk TMA. Explicit accumulator initialization removes
unnecessary initial stack traffic: SM90a frame528->56B at max64,272->64/72B at max32.
All12 configurations lose on H200 lab c8c89f9f111845af. Best synthetic-real M128554.0us vs V9292.5; M256835.7 vs543.2. Board-routing best833.2/995.0us vs280.4/453.4. Whole-kernel timings do not isolate stack, barrier or decode costs, so no causal mechanism is established. The local Humming incumbent below is not a valid real-decode control: exact-weight lab26d7e05ab97d4f96 rejects Humming/V11 agreement.

Only M128/256 acquire new per-shape dispatch entries. Original small-M winners and unknown
shapes retain V9_phase. Tuning embeds `moe_tune.py`'s real-router fixture exactly: if full
router/norm/pooler weights are supplied it uses them; standalone MoE lacks these, so it
generates model-distributed router weights with mean alignment, bias and the same stream.
It does not tune on uniform routing or inspect submission inputs. Three fresh fixtures,
128MiB float32 L2 flush, graph capture, four-pair coarse trials and20-pair finalists.
Admission uses an independent dequantized FP32 self-reference: global1.2%, worst row6%
(0.8 times MoE's1.5% and5x row bounds). Every optional phase catches any Exception and
logs rejection; a diagnostic failure leaves V9_phase. Deadline230s from prepare entry,
equal remaining time shares per shape; an in-flight JIT cannot be preempted.

## Local results, sm_120 / GPU0

Final official screen: **5/5 tests,6/6 fresh-input graph shapes**, **7.935s cached prepare**,
1.099x local overall versus SGLang. This is a passing screen, not a demonstrated board gain.
Both new shape choices won by less than1% on the synthetic real fixture; the official
normal-route timing is slower, so no H200/serving gain is inferred from this screen.
All36 compact lines /2549characters remain in the last16000 output characters.

Direct full-loaded-router benchmark, fixed max64/stages5/grid1/min16, median of20 cold
fresh graph pairs (raw CUDA events include replay overhead):

| M | Routing | Local pairs /experts /hot | V9 native us | Grouped us | Grouped/V9 paired ratio |
|---:|---|---|---:|---:|---:|
|128|real|312 /10 /108|403.54|402.48|0.9827|
|256|real|672 /8 /221|700.45|580.64|0.8244|
|128|matched uniform|312 /32 /18|710.75|1076.22|1.4965|
|256|matched uniform|672 /32 /31|1044.54|1161.20|1.1217|
|128|all-local uniform|1024 /32 /48|1214.51|1197.06|0.9855|
|256|all-local uniform|2048 /32 /85|10402.80|1626.11|0.1564|

GPU0 is shared. V9's M256 all-local uniform time varied4.8–10.4ms between runs; this
diagnostic's large ratio is not credited to H200. The actual V9_phase M256 incumbent is
Humming,563.30us on the real fixture versus580.64us grouped (paired ratio1.0476).
M128 is effectively parity; M256 beats native V9 locally but not that incumbent.

276 direct arithmetic/graph checks pass (12 configs x2 sizes x3 routing modes x3 banks,
plus60 edge replays at M1/8/33/129/257). Relative L2 about0.255%; all-local, all-remote,
duplicate/hot routes and shifted local_start pass. Strict signature stand-ins reach all24
shape/config Python ABI calls with132SM metadata; forced SM90a flags are cross-compiled.
Fault injection rejects configure, capture, numeric, timing, finalist and tuple failures;
expired budget and optional-prepare AttributeError keep the original incumbent.

Evidence: `runs/screen/tma_grouped/` (ignored):
`muse_moe_w4a16.cand_tma_grouped.final.json`, `screen_final.log`,
`validation_final120.json`, `tuner_safety_final.log`, `compile90_zero.log`,
`resources90_final.log`, `sass90_final.log`.

Reproduce with `CUDA_VISIBLE_DEVICES=0 tools/img python` and:

- `tools/local_eval.py problems/muse_moe_w4a16/dev/parked/cand_tma_grouped.py --json <output>`
- `problems/muse_moe_w4a16/dev/tma_grouped/validate.py`
- `problems/muse_moe_w4a16/dev/tma_grouped/tuner_safety.py`
- `env TORCH_CUDA_ARCH_LIST=9.0a python problems/muse_moe_w4a16/dev/tma_grouped/validate.py --compile90`

Do not regenerate the frozen candidate after handoff; future variants need a new name.
H200 execution/cold prepare/choices/speed, decode-layer integration, concurrent serving
replays and serving agreement remain unverified. The serving adapter needs per-lane
counter/workspaces, existing expert precision/norm semantics and EP collectives retained.

Precision audit (2026-10-24): raw17 banks retain packed NVFP4 bytes and E4M3 scale bytes, only permuted; no BF16/FP8 expanded weight bank is prepared. FP32 dequant in `_tg_reference` is diagnostic-only and not returned in state or read by timed kernels. `cand_v11_gemm` likewise retains packed/scaled expert weights in prepare and expands inside `_experts_dequant` on every timed call. The owned serving `muse_proj`, `muse_dflash`, `muse_prefill_gemm` modules handle BF16 dense weights and have no NVFP4 weight bank.

The inherited V9 prefix is different: `_encode_scales` rebiases scales into uint8
exponent/mantissa codes and folds sign/zero into FP4 nibbles. It is exact and has
no floating-point expanded bank, but it is not a permutation of the stored E4M3
bytes. Only the new raw17 bank passes that stricter literal byte-preservation
audit. This withdrawn candidate is not cleared for resubmission under the newer
scale-layout instruction; do not copy its inherited encoder into new work.
