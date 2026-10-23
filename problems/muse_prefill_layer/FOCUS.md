# Prefill focus
Serves: G2, G1.

- Rank 6 / P2; serving prefill is 753–16K tokens. Read `../../docs/FINDINGS.md` “Prefill evidence and serving path”.
- Best H200 receipt/source and per-load rankings are there; `submission.py` is the handler-promoted board source.
- Primary task: improve gate/up at753/3012 and grouped experts+dequant at16K; full64K FA3 is the largest individual chunk stage.
- Serving keeps one canonical V6 bank; contract: `../../docs/reference/expert_layout.md`.
- Active candidate: `cand_expert_portfolio_v2.py`, synthetic per-shape BN64/128/256, canonical pipeline and direct-bank DG choice.
- Complete stage combination: `cand_prefill_portfolio_v2.py`; fixed v11-style canonical control: `cand_canonical_hybrid.py`.
- Reset order, frozen SHA256 and exact local screens: `dev/NEXT_CHECKS.md`; orchestrator sends.
- H200 profile/floor detail: `dev/V11_PROFILE.md`, `dev/v11_profile_ranking.json`.
- Expert port: `srt/layers/muse_spark_moe_v4.py` (`prepare_layer`, `run_layer`, `_grouped_call`); keep finalize/EP contracts.
- Attention port: `FlashAttentionBackend.forward_extend`; mixed batches/head counts still need serving validation.
- Runtime evidence: judge receipts in `../../runs/muse_prefill_layer/`; handler state in `dev/finish_round_state.json`.
- Existing receipt handler may promote its original checked sources; do not duplicate accepted jobs.
- This round sends no remote jobs, starts no handler and makes no commits; work is solo per the current task.
- Remaining: H200 execution/selection, released-checkpoint memory and serving TP/EP agreement.
