# P3a operand and MMA issue assignment

**Measured, receipt `33f5c4c105374311` (exact S19, stage-only diagnostic; no candidate).** M8/16/32
conditional savings, us: MMA+activation 8.6/12.5/19.3 (modeled M8-32 six-load R1.0399
[1.0358,1.0441]); accumulator recurrence 4.5/4.5/6.9 (R1.0157); epilogue NOWAIT 4.4/5.3/6.8
(R1.0163); pack-free register 2.5/4.0/5.0 (R1.0110); shared fragment 2.4/3.3/3.2 (R1.0087);
useful-N MMA 3.1/1.2/1.3 (R1.0064); Bf register lifetime -0.4/0.0/0.9 (R1.0011). These overlap and
are not additive. The correct LAZY leaf loses (152.2 vs 96.7 us at M32). The S22 builds that priced
these terms all lose in the production graph: [builds22](../builds22/README.md). Full tables:
[REPORT.md](REPORT.md); `read.py <receipt>` reproduces `results.json`.

The batch separates shared/register pack-free BF16 fragments,Bf register lifetime,accumulator
recurrence,useful-N HMMA instruction count and epilogue pressure. It retains exact S19 BASE/AA,
sink/reservation controls,same264CTAs/depth2/tiles,eight original matched seeds×four symmetric
all-round arithmetic means. All1920 times,96NVML snapshots,work histograms/finish endpoints and
seed0 per-SM records survive. Diagnostic outputs are ignored;wrong-math cuts never nominate.

Source premises read: public MoE P4_HANDOFF.json/MECHANISMS.md (no transferable port winner),
and FINDINGS mixed_wgmma DIRECT/TRANS/WS/CTA2/CHUNK history. No donor code is copied.
S19 already loads BF16 bits directly;conversion-removal is not an arm. Per-CTA tile histograms and
32retained HMMA sites reconstruct issued work and useful-N equivalents,not hardware counters.
The PAD_USEFUL oracle skips K-warps to realize the count budget;it is not a narrower-N kernel.

All56 offline sm90a arm/shape rows fit two CTAs,zero stack/spills. ACC_CTRL/FREE both128regs atM8+;
live sinks keep all32HMMA (the first unretained version compiled to one). FRAG_SMEM117regs,
FRAG_JIT63,LAZY96 versus BASE126. Shared fragments add8320B,reserved by BASE_PAD too.
The8KiB synthetic repeated pattern omits actual packing and64KiB real working-set storage.

Exact native local replay passes448control TASK checks;LAZY separately passes162native checks on
M1/8/9/16/24/32,real/all/empty routes,local_start0/96/224 and changed inputs under reused captures.
Exact local packet8985B fits11700B. Local sm120 times are not H200 timings. Static/admission/import,
sm90a codegen and lab-only wrapper smoke pass. `admission_failure.json` records an earlier no-job
outer-wrapper rejection;the literal lab wrapper and a local stdlib-name collision were fixed before
the exact screens reran. No judge quota or measured recipe loss came from either failure.

Read on assignment:

```sh
python3 tools/h200_lab.py read <job>
python3 problems/muse_decode_layer/dev/nt1_timeline/structural/operand/read.py runs/<host>/<receipt>.json
python3 problems/muse_decode_layer/dev/nt1_timeline/structural/operand/report.py
```

Find the job by wrappere1710fb3 or the operand decision in runs/ledger.jsonl/runs/pending/.
The reader verifies wrapper/payload/cells/packet checksum,TASK controls,all runtime resources,
round/telemetry completeness and CTA work accounting. Invalid LAZY correctness remains unmeasured.
No failed global control becomes a measured loss. `--local-screen` audits only the archived sm120
screen and writes into ignored runs/screen/,never results.json.

Next: rank measured conditional contrasts by six-load ceiling and build against the largest feasible
mechanism. A real64KiB shared Bf cache plus two raw slots fails two-CTA capacity;one slot fits113808B
atM32 but needs a favorable bound after repeated LDS,publication and old one-slot cost. It is not
admitted yet. LAZY is a native-correct leaf probe,not a whole-layer build or winner.

`production_plan.json` pins frozen S21 expanded109c0135,shipmenteacc087b and its symmetric protocol.
The correct build must include timed packing/producer work and run all12 production graph shapes on
fresh disjoint seeds under promotion.evaluate. Whole-load T=46t_sliding+16t_full and held-out six-load
R lower bound>1 judge it;per-call time is not a transferred score. P4_HANDOFF.json currently has no new
winner;all83prior point nominations remain unchanged. No RULINGS entry is written.

Rebuild evidence only after preserving the queued snapshot:build.py,count.py in tools/img,
build_lab.py,local_eval --payload --tier static,run_local.py,check.py,read.py --local-screen.
sizing.py uses only5ee35b94's original measured counts/times and explicit planning assumptions.
send_batch.py refuses duplicate queue.json and passes --lane P3a --decision through h200_lab.
