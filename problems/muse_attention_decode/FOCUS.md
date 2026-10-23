# Attention verify

Serves: G2, G3.

The active candidate is `cand_attn_portfolio.py`. It keeps the scored q_len=1
kernel from the 1.39x submission and, for each scored q_len=8 shape, measures
all 27 FA3 `(num_splits, pack_gqa)` settings plus the eight K128 producer/MMA
and eight K64 producer/WGMMA variants from `cand_xqa_next.py`. `prepare()` uses
fresh scattered page-1 synthetic inputs, cold-L2 reads and CUDA-graph replays;
only the shape-keyed winner survives. Every measured time and agreement check
is printed as `PORTFOLIO_SHAPE`, with the selected path in `PORTFOLIO_PLAN`.

The local sm_120 screen passed all 16 correctness tests and 12 graph timing
shapes. Its timings are admission evidence only; H200 timing comes from the
orchestrator's receipt. On sm_120 FA3 is unavailable, so the fixed group XQA
stand-in is used for q_len=8 while the q_len=1 path remains unchanged.

`submission.py` remains the exact `32c5b4b287c24c38` source (1.390910x check,
SHA256 `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810`),
and its scored run is already queued. The verify goal remains open until a
receipt shows a custom or tuned verify path above FA3, especially full B16/B32.

The old `cand_xqa.py`, `cand_xqa_p8.py`, and `cand_xqa_ptranspose.py` sources
were moved under `dev/` because their XQA variants already have H200 coverage
through the prior diagnostic. Their builders and timing scripts now read the
private paths, so screen-all will not spend judge slots on duplicate sources.

Frozen hashes and the send order are in `dev/NEXT_CHECKS.md`. Only the
orchestrator sends or commits.
