# Attention verify

Serves G2/G3. The active next source is `cand_attn_verify_stream.py`; the
queued portfolios remain untouched.

The H200 q1 receipt `aff09765beaf429a` scored 1.3727x, below our 1.384x board
best. Decode remains strong (1.877-2.075x); verify is 0.952-0.995x because FA3
reads the same KV bytes as decode but keeps its 1.0x-level stream. Full verify
B16/B32 are the main opportunity; sliding verify is mostly fixed overhead.

The new stream kernel preserves q1's page-1 gather and pipeline, maps 8 draft
tokens × the GQA group to 64 MMA rows per KV head, dequantizes FP8 KV in
registers, applies exact causal masking, and merges split `(m,l,o)` partials.
Its portfolio keeps FA3, XQA/WGMMA, FP8-PV, and two stream variants per shape.
Failures are rejected without aborting; the judge's TASK.md 15% maximum is the
admission gate, not a 3% FA3 proxy.

Local sm_120: 16/16 tests and 12/12 graph shapes pass. H200 execution is
unverified. See `docs/FINDINGS.md` for receipt evidence and
`dev/NEXT_CHECKS.md` for frozen send order and hashes. The orchestrator sends
and commits.
