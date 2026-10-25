# Verify-load weight-streaming GEMM (P7 / G1)

Owned shapes: M64–256, plus the M8 diagnostic. The projection lane owns M1–32
and glue fusion. This block changes only BF16 dense GEMMs; it retains no
dequantized NVFP4 weights and never reads an expert scale bank.

`kernel.cu` puts 64 token rows on WGMMA's fixed M side and 64/128/256 weight
rows on its N side. One producer warp issues weight-first TMA requests into
2–8 slots. One, two or four consumer warpgroups reuse the staged weights.
The pipelined variants keep one committed WGMMA group outstanding and release
the preceding slot only after `wait_group 1`; the final group drains before
stores. The other variants wait for each group. K64/128/256 transactions and
split-K target one/two waves of 132 H200 SMs. A release/publication fence and
last-CTA FP32 reducer finish split-K in the same launch, resetting private
counters before exit. There is no cross-CTA spinning or timed initialization.

The canonical SW128 descriptor is shared with the earlier validated TMA
reader. Local TMA and the fragment loads execute; `mma.sync` stands in for
WGMMA on sm120. The local ring/K-stage shrinks when necessary for its 99KiB
shared-memory limit. The native build retains the full ring and K stage;
local success does not establish native execution or speed.

The complete M128/256 bank is **197132288 bytes**, not 138MB: qkv37.75MB,
o16.78MB, pool/router75.50MB and unpool67.11MB. M8's sliced unpooler makes its
bank138412032B. At M256, the former performs50.466GFLOP: one read at4.8TB/s
is41.07us, math at800TFLOPS is63.08us (51.03us at989TFLOPS).

## Decode dispatch

`python3 build.py` emits `projection.py` and `../verify_streaming_lab.py`.
Inline `projection.py` in the decode candidate. In prepare, call
`vs_prepare_dispatch(state, recipes)` with receipt-selected
`{(tokens, role): (code, split_count)}` entries. Set the corresponding
`state['proj_plan']` entries to `('verify_stream', code, split_count)`.
At the start of `_project_variant`, dispatch that tag to
`vs_project(x, w, name, state, out_dtype)`.

The return is exactly `(contiguous_output.unsqueeze(0), 1)`. QKV/O/unpooler
outputs are BF16. Pool/router preserves all4608 FP32 rows (4096 pool,256 router
hi,256 router lo); the existing next consumer adds the hi/lo rows. M8 uses
`unpooler_slice`; larger token counts use `unpooler`. No selected recipe is
installed before H200 evidence. Each instance owns its output/scratch/counters;
separate concurrent streams require separate instances.

`build_fixture.py` generates the local-only full-layer fixture, replacing
all twelve projection entries at M8/128/256 with real portable operators.
The board candidate is unchanged. This is separate from best7's existing
`streaming` setting, which currently refers to the projection lane's GEMV block.

## Lab and evidence

`../verify_streaming_lab.py` uses the host's real loaded BF16/router weights,
three fresh fixtures (normal, outliers, single-column probes), FP32 and per-row
gates, FP32 router-hi+lo agreement, and replay-counter checks. All407 eligible
recipes are screened; the top two per key face the control for20 rotated cold
graph replays, flushing256MiB each time. M128/256 run before M8. Raw graph
times and paired ratios are reported; they include graph overhead. The TOTAL
rows sum four standalone graph medians, not an integrated layer measurement.

Controls are the actual best6 `1b80ffc037cc454c` choices, extracted from
`cand_r3_best6.py`: M128 QKV NN23 and pool/router TN0; M256 QKV TN0; cuBLAS
elsewhere, except best6's M8 pool/router and sliced-unpooler split partials.
Those partials are reduced only outside the timer for correctness, as their
production consumer does that work itself. Every simultaneously live graph
has its own native capture stream: pooled streams reproduce a finalist-replay
launch failure; workspace lifetime is the inferred mechanism. Distinct native
streams fix the full reproducer, and the corrected full lab passes.

Local admission: **744** kernel checks, **90** fresh-input graphs, **415** native
descriptor constructions, **3** ABI rejection cases; full lab **407** passing
recipes/12 keys, zero rejections. The forced layer passes **13/13** tests,
**12/12** scored graph shapes and **15/15** static checks, prepare11.36s.
SM90a: **15** kernels, no spills, max154 registers, HGMMA/TMA in SASS.
`validation.json`, `lab_validation.json`, `integration.json` and
`compile_sm90a.json` preserve the checks. Local timings do not select winners.

The prefill-host screen also passes **5/5** tests, **8/8** scored graph shapes
and **3/3** static checks, prepare18.40s (`host_screen.json`). The send merged
the waiting projection-glue payload into the same run: send key
**be5d6aa5ae1ac0bd**, judge job `5535d2f171b34d72`: weight-wide WGMMA loses 1.28-2.18x
(FINDINGS). Payload SHA-256:
`7b67602f1d4b77c0ccb074d52772d93e7fd228767900fb07178565373896445d`.
The two payloads get separate namespaces, output shares and time budgets.

Reproduce from the repository root with `CUDA_VISIBLE_DEVICES=0 tools/img
python problems/muse_bf16_proj/dev/verify_streaming/validate.py`; use the same
prefix for `validate_lab.py` and `audit.py`. `read_receipt.py <job>` emits the
decode handoff only from a matching, complete native-H200 local receipt.
