# Closed-recipe score-gate audit

Audited 63 FINDINGS rows and 90 primary receipts dated 2026-10-23 onward. Source predicates and available raw arrays are indexed in `inventory.json`; all row verdicts and exact statistics are in `rejudgment.json`.

Per seed, R is the six-load geometric mean of old/new, with 46 sliding + 16 full per load. Intervals use the shared score gate and A:A noise. Archived component/profile repetitions are not substituted for paired input seeds. No host-equivalence or exact-ID criterion remains a promotion condition.

| Recipe / receipt | Old decision | New timing R [95% CI] | New decision |
|---|---|---|---|
| register preloads, `0f74021a`/F | closed / per-load veto | 0.887139 [0.885708, 0.888572] | timing loss |
| hints + metadata, `0f74021a`/B | closed / per-load veto | 1.002694 [1.002068, 1.003320] | timing pass; nominate |
| metadata, `0f74021a`/M | closed / per-load veto | 1.003169 [1.002493, 1.003846] | timing pass; nominate |
| d1-full glue, `0f74021a`/D | closed / per-load veto | 1.002841 [1.002001, 1.003683] | timing pass; nominate |
| hints + metadata, `8db42132`/B | closed / per-load veto | 0.885121 [0.883630, 0.886615] | timing loss |

B/M/D used 16 held-out inputs `2100911 + 104729*i` on S14 synthetic weights seed 5353 / router 197731. They passed the old lab host-agreement checks. Those checks are not public TASK; no full promotion is asserted. D is the prior d1-full glue policy, not a combination of B and M. Rank source-wide policies by R: M, D, B, F.

Shape nominees: q1 full page-flight (B16/B32); heavy LIW/TYPW and M256 PAIR2/adaptive candidates against current PERMW. Their receipts retain means/ranges rather than six-load seed pairs, so exact G0 CI is unavailable. DSM and register-fold winners are already represented in S16/S17. B16 pair-register V loses to the measured N128 comparator even though it beats the older S13 point.

Reopened correctness: in-kernel own-weight prefetch `7426e17c` and the old persistent-qkv recipe. Their host-error / exact-ID rejection does not establish a TASK failure. Missing timing stays unmeasured. The higher-impact decode-once/producer-consumer, queue, persistent and TMA recipes retain their large measured losses; relaxing a gate does not make those samples faster.

Reproduce: `python3 problems/muse_decode_layer/dev/gate_audit/inventory.py` then `python3 problems/muse_decode_layer/dev/gate_audit/rejudge.py`. The compressed archives come from immutable `runs/` JSON; generated extracted sources/logs stay ignored.
