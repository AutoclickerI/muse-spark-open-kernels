# Routing, pre-expert norm and metadata

The cooperative repair and its H200 lab `9eeb39ba` (lost) are in [coop_glue](../coop_glue/README.md).
`route_once.cu` computes top8/bias and pre-expert RMS once per row, publishes
IDs/weights/inverse norms through a cooperative grid barrier, then retains the
packed expert MMA body. No BF16/FP8 expert bank is prepared. M256 retains masked
Humming's original packed bank.

`once.py` derives it from the tested private-CTA source `route_moe.cu`; `rg_value`
converts FP32 pooled activations to their normalized BF16 values at expert loads.
A per-call BF16 placeholder allocation is reused as FP32 inverse-norm scratch;
it is never an activation or a prepared weight bank. All scheduling is replay
safe and shape driven. The active standalone cell is
`../coop_glue/best7_coop_glue_cell.py`, with its dispatch table empty.

`MoERouteMeta` v1 is shared with `../nt1_branches/route_meta.h`: 2944B/736int32,
capacity32 tokens and256slots. The producer writes version, tokens, EP start,
per-expert counts/offsets/blocks8/unit prefixes, token-local counts, expert-major
pair list and inverse slot map. Every word, including padding/remote slots, is
rewritten each call. `rg_expert` returns `(out, ids, weights, local_ids, meta)`;
metadata is empty above32 tokens. The expert body still reads its private summary;
consumer integration belongs to the small-M lane. The active batch checks all
736words in24 graph replays, beside180 full-layer comparisons with zero flips.

The earlier private-CTA metadata producer passed84 exact descriptor checks
(`check_metadata.py`) and108 forced-Hopper whole-layer comparisons, zero flips.
`metadata_admission.json` pins that version. Its `build.py --dropin-only` rebuilds
the old disabled standalone cell; the cooperative variant is built by `once.py`
and `../coop_glue/build_cell.py`.

Receipt `8a3d71811a324c36` used the older source without metadata and returned no
timings: after74s, one M256 sliding pool-epilogue route flip raised an exception.
Its source is frozen in
`runs/muse_moe_w4a16/20261024T154402Z_check_8a3d71811a324c36.py`;
`admission.json` refers to that immutable source, not the rebuilt payload.
Payload builders now record flip details, disable that cell/M and continue. The coop
lab `9eeb39ba` reran the old prefix: the flip is two experts swapping slots inside
the same top-8 set (cutoff loss 0), not a routing error (FINDINGS "Failures").
