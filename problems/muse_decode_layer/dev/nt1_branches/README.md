# Small-M MoE: shared routing and fixed-phase experiments

Owns M1/8/16/32 for the overall decode-layer score. H200 parent dc3fea0a landed:
one-LOP3 NT1 wins42.74/73.75/79.20/113.61us vs R2 43.08/74.45/81.21/117.22.
The exact admitted cell is `nt1_lop3_cell.py`, SHA
`3f84a8a4e12b41bd7257712d8656d3f417f6a9e238f00e481b3be526da69cc40`.
`lop3_bind(existing_recoded_bank)` aliases its packed weights and counters;
`lop3_experts(x,wts,local_ids,cell_state)` serves M<=32. Decode FOCUS owns
integration. The121-register `lop26` build is a separate unmeasured control;
the handoff preserves the H200125-register CUDA exactly.

The widened-scale fold25 recipe is retired as inadmissible. All current banks
keep four-bit weights and one-byte scales; sign/zero recoding is lossless.
`fixed_cell.py` prepares only one selected bank/variant. New variants are
experimental and do not change best7.

## Metadata ABI for the glue lane

[route_meta.h](route_meta.h) defines `MoERouteMeta`, aligned128,2944 bytes.
Every field is int32; offsets below are int32 indices, not byte offsets.

| Field | Offset | Elements |
|---|---:|---:|
| version=1, tokens, local_start, local-pair count |0|4|
| expert counts |4|32|
| exclusive expert offsets, offset[32]=pairs |36|33|
| ceil(count/8) token blocks |69|32|
| W13 unit prefix =512*cumulative blocks |101|33|
| W2 unit prefix =256*cumulative blocks |134|33|
| local slots per token |167|32|
| sorted position -> token*8+slot |199|256|
| token*8+slot -> sorted position; -1 remote |455|256|

Only `tokens<=32` and `tokens*8<=256` are admitted. Sort by local expert, then
original flat slot index. Remote IDs are excluded. Expert identity is ID minus
`local_start`; layer-local IDs use local_start0. Unused list entries are-1;
padding has no meaning. The metadata is input-derived and written every timed
call, never in prepare. `route_meta.cu.inc` is the standalone one-CTA producer.
A fused router can emit this format directly and omit that extra launch; the
MoE main kernel consumes `const MoERouteMeta*` as its final argument. It reads
prefixes to schedule work and copies the slot maps, without re-scanning top-k.
The current A wrapper includes the standalone metadata launch in its timing.

## Composable changes

`build_moe_nt1_fixed.py` assembles compile-time switches from the admitted LOP3
CUDA. A/B/C are independently selectable and combined in ABC; D is measured
alone and combined at M16/32.

- A: one metadata pass, then CTAs copy the shared routing description.
- B: retain the existing expert-local W13 ready counter; remove global norm
  zero-init waiting and the final grid barrier. W13's first row tile clears
  each expert's norm accumulators before release publication. Expert-local
  W2 completion elects a CTA to normalize its slots, then per-token release/
  acquire tickets elect the vector combine. Counters reset within the call;
  remote-only tokens are explicitly zeroed.
- C: all CTAs request their assigned W2 cache lines into L2 while W13 runs.
  Existing loads already use bulk TMA and128-bit shared vectors. This is timed
  lookahead, with no persistence setting. K2 separately tests more CTAs.
- D/D128:128 producer threads decode packed microtiles to double-buffered
  BF16 shared memory;256 consumer threads run m16n8k16 MMA. Microtiles contain
 64/128 K elements per consumer warp; raw stages are4608/9216B and decoded
  stages16/32KiB. Stage2 totals53,384/95,368B at M32. Same-size prepare-time
  layout permutations only; every dequant operation is inside the timed call.

Source inspection corrects the earlier barrier interpretation: NT1 already
waits on each expert's W13 completion. Its grid barrier is after W2, before
combine; a separate global wait protects norm zeroing. CTA phase medians
are nonadditive and do not establish recoverable end-to-end savings.

D uses `[fragment][warp][lane][4 regs]` shared layout. The initial layout's
16-word lane stride caused four-way conflicts within128-bit access waves;
the4-word stride gives one use per bank per wave. D128 additionally halves
barrier/prefetch iterations. WS copies use normal L2 policy; the incumbent
uses evict-first for single-block experts, so D includes that policy difference. Main-kernel sm90a registers A/B/C/ABC are
126/119/125/119; D/ABCD70/72, D128/ABCD12878/79. All have no spills/local ops.

## Lab, evidence and status

One MoE-host payload `prof/lab/moe_nt1_fixed.py` compares R2, admitted INC,
L26, A/B/C/ABC and K2 stages2/3 at M1/8/16/32, plus D64/D128 and their all-
combined variants at M16/32. Twelve public model/router seeds, identical
real routes per arm, four cold128MiB AB/BA pairs per arm, independent INC A:A,
FP32 gates before/after timing, and exact non-K2 SwiGLU/W2 stage comparisons.
First failures print expert/pair/row/channel. Compact per-M means print last.
The wrapper deadline is reduced5s so the host stays within280s.

Admission:1008 native graph gates,1584 bit-exact intermediate comparisons,
288 metadata checks,864 forced Hopper calls; signed/zero/subnormal scales,
empty/all-local routes, zero inputs, start96, fresh replays and reset counters.
Max FP32 relative L2 is0.242%; output differences from INC are confined to norm
accumulation/rounding (max relative0.030%). Standard/WS128 stage3 glue uses
native stage2 stand-ins on SM120; WS64/K2 native depths1–3 are exercised.
All12 source families/20 sm90a entries compile without spills/local loads.
The host wrapper passes5 public tests/6 static checks; its full12-seed payload
finishes in7.2s locally, prepare9.3s. Local new-path rankings mostly lose;
H200 rankings remain unverified. See [admission.json](admission.json).

Sent once: **d52b02e0bc91d390**, accepted job **383ccc1fbbae41e2** on MoE.
The immediate detached log confirms acceptance and no rejection. No polling.

```bash
python prof/lab/build_moe_nt1_branches.py
python prof/lab/build_moe_nt1_fixed.py
CUDA_VISIBLE_DEVICES=0 tools/img python prof/lab/check_moe_nt1_fixed.py
CUDA_VISIBLE_DEVICES=0 tools/img python prof/lab/check_moe_nt1_handoff.py
```
