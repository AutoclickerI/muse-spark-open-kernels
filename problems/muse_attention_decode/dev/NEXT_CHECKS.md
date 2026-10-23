# Attention checks at 09:00 KST (2026-10-24)

The owner leaves sending to the orchestrator. Do not launch new `okc_run.py`
processes from the attention lane. Do not resend a source already accepted.

| Order | File | Frozen SHA256 | Action |
|---|---|---|---|
| 0 | `cand_xqa_diag.py` | `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Its existing sender is waiting locally for admission (priority 0); preserve it. Read `XQA_DIAG`/`XQA_PLAN` when its receipt lands. |
| 1 | `cand_xqa_k128.py` | `b519462266ffbfd9379979cd59891fecbc6b12c9f563d9c3ba8009a0efa4730f` | New packed eight-warp K128 diagnostic: 24 configurations plus FA3 at each verify shape, full B16/B32 first. One check covers all configurations. |
| 2 | `cand_xqa_pipeline.py` | `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` | Private K128 full-specialized pipeline: no runtime RoPE branch on full attention, 3-stage 48 KiB gather, waves 2/4/8 plus paired FA3. New local screen passes; leave sending to the orchestrator. |
| 3 | `cand_verify_sweep.py` | `2f0205924f1b6ea5f16d27a9b0a5465b6374144987f01d00ad3344da69d6605e` | Already waiting through screen_all; collect it, do not launch another. |

Sent scored XQA sources remain frozen at these hashes: `cand_xqa.py` `551949b5eb2219d0fd698ab202e529d21095eb08edb8171701796067538f7296`, `cand_xqa_ldm.py` `3511f6ae5cbec225a60cf04696b766e729bd46c317e22c65d3a7382efe285ee9`, `cand_xqa_p8.py` `5b856076f62f9402d4d1e2f2bfc02efd1a12def8090321d35748fb24a8f9df43`, and `cand_xqa_ptranspose.py` `308dbb4fba8e13d22654881227d69596847005a2b7813c39c6af136f22872f7`.

The gather experiments are closed for this round: register offsets were neutral, shared offsets slower, and the bulk ring hung q8. They are not candidates or queue entries. Preserve their logs as evidence and wait for the already sent XQA receipts before changing dispatch.

The sent K128 file replaces the old spilling four-warp source. It preserves the
q_len=1 implementation. `prepare()` sweeps FP16/FP8 PV, 3/4/5 gather stages and
1/2/4/8 scheduling waves, prints `XQA_W8_DIAG`, then compares the best custom
variant to FA3 in swapped order (`XQA_W8_PAIR`). A custom per-shape dispatch is
selected only above 1.03x in that paired comparison; otherwise it uses FA3.

Admission evidence for the exact SHA: `dev/xqa_w8_final_local.json` has 16/16
tests and 12/12 graph shapes (three replays, two fresh input sets). Additional
checks: `xqa_w8_formats.log`, `xqa_w8_variants.log`, `xqa_w8_glue.log` and
`xqa_w8_sm90_compile.log`. All six sm_90a specializations have zero spill loads
and stores. FP16 stage 5 needs 104 KiB shared memory and is checked locally
through a stage-3 CUDA stand-in at the same Python ABI; the other five stage /
PV combinations execute their real kernels locally.
The exact final JSON/log are copied under `runs/screen/attention_w8/` with the
standard names; `screen_all.earlier_pass()` recognizes them and can reuse them.

The pipeline candidate has no H200 job ID; it is locally screened and left for
the orchestrator. The old
diagnostic's `runs/screen/attn_diag/send.log` confirms local admission waiting.
Inspect `runs/pending/` and `runs/muse_attention_decode/` before sending.

Promote an exact passing source to `submission.py` only if its H200 check beats
1.388x, then let the orchestrator send the scored run. Keep full B16/B32 above
FA3 as the primary decision, followed by sliding verify. Current submission
remains the 1.384x hybrid (`526f31018e2c4449`, check `4f8c41ef9e924ff4`).
