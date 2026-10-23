# Active prefill round

Current snapshot: 2026-10-23T14:23:34+00:00. This file is rewritten by the receipt handler; experiment history is in runs receipts and RESULTS.md.

Owner instructions: the two-failed-ideas stop condition was revoked. Do useful development while judge jobs run. Edit only this problem; the original no-commit hold remains. Keep accepted candidate sources immutable. Use tools/okc_run.py for every remote check/submit, with a note. Count accepted and waiting checks before sending; the external limit is20/day, resetting00:00UTC/09:00KST.

Current submission.py: check82a4e50b54294a2a, 1.345393014x. SHA256 670b945489abc963b19ebe5581dae6be79dec35dffca6e5ce17035252ca15858. Last verified board: rank55, 1.2124x. Check performance and board performance are distinct; pending scored runs do not establish a new board result.

Accepted this UTC day: 15/20 checks, 2/10 scored. Prefill check admission requests: 0. Accepted plus waiting checks: 15/20. Final integration status: awaiting component receipts. Missing component receipts: [9, 10, 11, 12, 13, 14, 15, 16, 17].

Pending accepted jobs:
- check 583a1b97cc544f53 (problems/muse_prefill_layer/cand_v10.py)
- submit bab2f3468fcb4b58 (problems/muse_prefill_layer/cand_v3.py)
- check 53b79ddde4a048d5 (problems/muse_prefill_layer/cand_v12.py)
- check f13e757f4c224e06 (problems/muse_prefill_layer/cand_v9.py)
- check 362f63065c634281 (problems/muse_prefill_layer/cand_v13.py)
- check 6a9cd943fe404d74 (problems/muse_prefill_layer/cand_v11.py)

Background result handler: dev/finish_round.py, PID1902861, logdev/finish_round.log. It watches actual receipt events, promotes exact passing checked sources, scores through the wrapper within quota, excludes failed scored reruns and rolls back if necessary. Each promoted source is frozen. A single-instance lock prevents duplicate handlers. It exits when this round's checks and scored jobs finish. Collectors and admission wrappers are independent processes; do not relaunch accepted jobs.

Final integration builder: dev/build_measured_combo.py. It selects experts by token-size bucket, attention by token/window buckets, and native dense from correctedv13 measurements. It statically prefixes private expert definitions; submitted code imports no other candidate and does not execute generated Python or start processes. The final check is sent only after official local tests, fresh nearby graphs, forced first-turn local tests and Hopper Python-glue stand-ins pass. The assembled research state may retain multiple weight layouts; share a layout or choose one family before full-model serving.

v14–v17 all passed5official local tests. v15 shared-A gate/down emulation passes; v17 covers every finite signed E4M3 scale and all FP4 codes with0.159% local GEMM relativeL2. v15/v17 sm90a builds have no spills. Hopper TMA flags forced onto unsupported sm120 failed; the actual guardedv14 passes. Actual FA3/WGMMA/TMA performance is verified only by H200 receipts. Combined and forced-first-turn rehearsals pass local layer/graph/glue checks; these establish no H200 speedup.

Sources and current evidence: RESULTS.md, PORTING.md, dev/results_summary.json/txt. Builders refuse source overwrite. Upstream sync in own-directory staging found official prefill files and llms unchanged. No serving edits or commits were made.

Next hypotheses when check capacity is available: fast decode plus256-token tiles if both measured components win; a direct packed-bank dequantizer for a memory-efficient custom/DG serving hybrid; lower-register/shallower-pipeline BN64 occupancy tuning if turn1 remains expensive. None is a measured H200 gain yet.
