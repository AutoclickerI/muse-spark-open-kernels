# Muse Spark Open Kernel Challenge

Our entry in <https://openkernelchallenge.dev/>: the fastest Muse Spark Open decoder layer (256 experts, NVFP4
weights) on one H200, scored on the overall kernel board. Goal, rules and plan: [AGENTS.md](AGENTS.md) and
[docs/](docs/). Organizer rules: `upstream/llms.txt`, `problems/<name>/TASK.md`. Serving is parked until its round
([docs/serving/](docs/serving/README.md)).

| | PT | KST |
|---|---|---|
| kernel track closes | Nov 4 23:59 | Nov 5 16:59 |
| serving track opens / closes | Nov 5 12:00 / Nov 19 23:59 | Nov 6 05:00 / Nov 20 16:59 |
| daily caps reset | 00:00 UTC | 09:00 |

Each kernel problem takes 10 scored runs and 20 checks a day (live: `okc problems`).

## Layout

```
problems/<name>/   okc init's files (TASK.md, reference.py, baseline.py = SGLang, checks.py) and ours:
                   muse_decode_layer/cand_r3_best7_s14_final.py (incumbent), cand_r3_best7_s15.py (next),
                   dev/<lane>/ (each lane's live work; its README has the handoff and next step)
upstream/          llms.txt, terms.pdf, the public API's JSON
tools/             okc_run.py (every remote run, receipts), h200_lab.py (H200 labs inside a host run),
                   local_eval.py (local screen), img (run in the judge image), sync_upstream.sh, board_model.py
runs/              receipts of every judge run (ledger.jsonl + per-run JSON), board snapshots
sim/, prof/        bounds model, judge-profile and SASS analysis
serving/           parked until the serving round
docs/              GOAL, CONSTRAINTS, PLAN, FINDINGS, SOTA, RESOURCES, RULINGS; reference/
```

## Kernel loop

```sh
tools/img python tools/local_eval.py problems/muse_decode_layer/cand_r3_best7_s15.py --tests-only
python3 tools/okc_run.py send <file> --note "what changed"        # scored while any left, else a check
python3 tools/h200_lab.py send <payload.py> --host <problem> --note "..."
python3 tools/h200_lab.py read <job>
```

Locally (sm_120, in the judge image) every problem's tests run; SGLang's FA3 baseline is Hopper-only, so attention,
decode-layer and prefill time the submission alone, and Hopper-only kernels (WGMMA, TMA) need forced stand-ins
(AGENTS). Only the judge times the H200. Setup and machines: [docs/RESOURCES.md](docs/RESOURCES.md).
