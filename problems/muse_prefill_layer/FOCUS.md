# Prefill focus
Serves: G2, G1.

- Rank6/P2; serving prefill is753–16K tokens. Durable evidence: `../../docs/FINDINGS.md` “Prefill evidence and serving path”.
- H200 control: v11 check6a9cd943fe404d74=1.367883x; scored965a9a816d4744ea=1.366811x. `submission.py` is unchanged.
- Active source: `cand_score_portfolio.py`; one prepare-time H200 sweep chooses expert, dense/router and attention paths per shape.
- Board layouts are unrestricted resources: raw, original WGMMA, bit-plane, canonical and Humming/Marlin banks coexist losslessly.
- The canonical V6 contract remains a serving interface, not a bank-count constraint. Allocate duplicates from measured throughput and KV need.
- KV scenarios and source flags: FINDINGS; reproducible arithmetic `dev/kv_budget.py` / `kv_budget.json`.
- Next send order, frozen hashes and exact local screens: `dev/NEXT_CHECKS.md`; the orchestrator owns sends and commits.
- H200 profile/floor detail: `dev/V11_PROFILE.md`, `dev/v11_profile_ranking.json`.
- Expert port: `srt/layers/muse_spark_moe_v4.py` (`prepare_layer`, `run_layer`, `_grouped_call`); preserve finalize/EP contracts.
- Dense port: `bf16_gemm_dispatch`; attention port: `FlashAttentionBackend.forward_extend` after KV writes.
- Humming heuristic configs must be copied before setting variant flags; cached dict mutation corrupted the local TMA stand-in.
- `dev/finish_round.py` was stopped/deleted. No check-then-promote daemon; `send` admission and board best retention own that workflow.
- Receipts: `../../runs/muse_prefill_layer/`; `dev/write_report.py` renders receipt JSON without launching work.
- Remaining: H200 selection/execution, held-out nearby shapes, released-checkpoint allocations and serving TP/EP agreement.
