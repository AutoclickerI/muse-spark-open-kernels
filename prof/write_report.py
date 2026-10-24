from pathlib import Path
import json,datetime,math
R=Path(__file__).resolve().parents[1]
d={x['problem']:x for x in json.loads((R/'prof/report.json').read_text())}
a=json.loads((R/'prof/analysis.json').read_text());manifest=json.loads((R/'prof/evidence_manifest.json').read_text())
out=[]
def w(s=''):out.append(s)
def link(p):return f'[{Path(p).name}]({p})'
w('# Profile and critical path')
w()
w(f"Evidence snapshot: {manifest['utc']}. Analysis lane only; no kernel edits, remote check/submit, or commits. Best means highest complete, passing, unflagged receipt, including checks; it does **not** mean a scored board entry. Source selection and hashes: {link('prof/evidence_manifest.json')}.")
w()
w('The immediate priority is the **verify MoE path inside the full decode layer**, followed by prefill expert GEMMs and the replicated prefill projections. Attention v2 helps the kernel board’s one-token loads but regresses every verify shape. A decode-only tuning result is insufficient evidence for a serving port. The whole-layer verify path still uses Marlin; the isolated best MoE has not yet been incorporated into that measured layer.')
w()
w('## Reproduce and interpret')
w()
w('```sh\npython3 -I -S -B tools/profile_report.py all\npython3 -I -S -B tools/profile_report.py muse_decode_layer --serving\npython3 -I -S -B tools/profile_report.py all --json\npython3 -B -m sim --offline\npython3 -B -m sim.serving\npython3 -B -m sim.calibrate\n```')
w()
w('The report tool is host Python with stdlib dependencies only, including the local `sim` modules. It performs no network or GPU work. Full saved per-kernel diffs, including each retained kernel row, are in [all loads](prof/report.md) and [serving loads](prof/report_serving.md); [JSON](prof/report.json) retains shape-level inputs. The selection is automatic on each invocation. Regeneration helpers are under `prof/`; generated caches and outputs are ignored there.')
w()
w('- **H200 measured / judge**: `result.records` benchmark `median_ns`, `baseline_ns`, load weights. Speedup uses the paired baseline in the same receipt, rather than dividing two independent profile totals.\n- **H200 measured / profile**: GPU kernel durations, rounded to 0.1 us in the saved report; their sum can include overlapping PDL waits. A profile sum is neither a paired graph median nor exclusive critical-path time. “GPU busy” is retained separately.\n- **MODEL**: bytes / 4.32 TB/s and BF16 math / 741.75 TFLOP/s from `sim.hw`, plus the design’s launch costs for total practical time. `% of bound` means 100 × modeled work time / measured profile time, **not** time divided by the bound. Practical rates are design targets, not rigorous lower bounds.\n- **STATIC sm_90a**: compiled locally, never executed on H200. **LOCAL sm_120**: independent local timing evidence; no conversion to H200 speed.')
w()
w('The judge saves only its top kernel rows and an “N more” count. Missing names and work attribution cannot be recovered from these artifacts. The tool preserves their time as `unattributed`, and emits `n/a` plus `unexplained` instead of manufacturing a per-kernel bound. Multiple cuBLAS shapes sharing a truncated name get a component bound once, not an arbitrary time-proportional allocation. This is the limit on the requested “each kernel” accounting. For identifiable fused MoE/attention/projection kernels the tool does provide individual efficiency and class. The classes are hypotheses: memory near SOL at ≥70% of the practical work bound, compute at ≥50% when math dominates, latency/launch when the small modeled memory work cannot fill a few us per call; otherwise unexplained. Efficiency >115% signals model/profile disagreement.')
w()
w('Upstream was refreshed by running a scratch copy of `tools/sync_upstream.sh` under `prof/upstream_sync/`, preserving the edit restriction. All downloaded official problem files and stored upstream files compared identical: [comparison](prof/upstream_sync/diff.json). Stored rules and receipts remain the primary sources; no login or rules acceptance occurred.')
w()
w('## Kernel boards: all scored loads; serving: verify and prefill')
w()
w('The whole-model load weights are 62 dense calls, 46 sliding + 16 full layers, and 60 NVFP4 expert layers. Tables below use those weights. They are step/load totals, not individual-layer times. The overall board’s exact aggregation is not established by these receipts; the decode-layer/MoE priority is supported by their time contribution and `CEILING.md`’s leaderboard-fit evidence, not a proved scoring formula.')
w()
for p,v in d.items():
 o=v['ours'];s=v['sglang'];w(f"### {p}");w()
 w(f"Best H200: `{Path(o['source']).name}`, {o['speedup']:.4f}×, CI {o['ci'][0]:.4f}–{o['ci'][1]:.4f}; {link(o['receipt'])}. Calibration: {link(s['profile'])}. Full kernel diff: [report](prof/report.md).")
 w();w('| Load | Ours judge ms H200 | Paired stock ms H200 | Speedup H200 | Ours / stock profile ms H200 | Practical ms MODEL |')
 w('|---|---:|---:|---:|---:|---:|')
 for l,t in o['loads'].items():
  z=s['loads'][l];w(f"| {l} | {t['judge_us']/1000:.3f} | {t['paired_baseline_us']/1000:.3f} | {t['paired_baseline_us']/t['judge_us']:.3f}× | {t['profile_us']/1000:.3f} / {z['profile_us']/1000:.3f} | {t['practical_us']/1000:.3f} |")
 w()
 if p=='muse_bf16_proj':
  w('Time is dense GEMM/router throughout. The best tested whole candidate is still slightly below stock overall; there is no measured whole-projection win to port unconditionally. Top three bottlenecks:')
  w('\n1. **Router’s high-token skinny tile**: at verify32 its profile is 28.7 us versus a 2.7 us judge floor, while qkv/pooler/unpooler cost 20.4/28.9/24.8 us. The same workload reaches only 0.885× overall. Router hi+lo preserves precision; its NT8 specialization uses 220 registers (STATIC).\n2. **Low-M qkv streaming/tail**: verify1 qkv is 16.3 us for ~37.7 MB, versus a ~8.8 us practical work target. There are only 144 row CTAs at the documented WN1/TR1 config; occupancy/tail is a hypothesis, not a measured counter diagnosis.\n3. **Replicated pooler/unpooler**: each full matrix is ~67 MB per rank per layer. At prefill their redundant math matters more than a marginal standalone skinny win; `docs/serving/serving_levers.md` proposes row split. Even the best sweep loses verify16/32. Keep the faster operations selectively; do not extrapolate the verify1 1.277× across token counts.')
 elif p=='muse_attention_decode':
  w('One fused `attn_main` occupies 100% of our profile; stock uses FA3 main/combine plus prep/norm/store kernels. Shape-level examples below are from the same-code v2 check [51ad2ea9ec4f4724](problems/muse_attention_decode/profiles/cand_v2.51ad2ea9ec4f4724.json); the newly collected scored rerun is selected in the table above. Top three bottlenecks:')
  w('\n1. **Verify arithmetic/occupancy**: sliding verify32 is 51.1 us in v2; full verify32 is 420.2 us. Every paired verify shape loses (ratios in `docs/serving/serving_levers.md`: 0.60–0.93×), while decode load gains are ~1.9×. Static verify register use is 230 versus decode’s 110, without allocator spills.\n2. **Full-context gathers at concurrency**: decode32’s fused profile is 3.737 ms per step against 2.610 ms of practical work; verify32 is 9.074 ms against 2.640 ms. The extra verify work reuses keys but requires more QK/PV and softmax operations; the simple bandwidth model is not the whole explanation. Page-size-one irregular gather efficiency is unmeasured.\n3. **Sliding fixed cost and split/cluster finalization**: verify1 sliding is 17.6 us for ~0.1 us of ideal data time. Source/SASS include barriers, shared partials and cluster/global-counter waits. One kernel does not eliminate its internal synchronization cost. Keep FA3 for verify until a replacement wins on H200; prioritize reusable verify tiles over further q_len1 specialization.')
 elif p=='muse_moe_w4a16':
  w('New evidence during this audit: v6 (`eefecf1eeb724feb`) supersedes the requested v4. Both are one fused kernel, so there is no remaining CUDA launch chain in this isolated profile. Top three bottlenecks:')
  w('\n1. **Resident pipeline capacity**: v6 changes five shared stages to two, NT2 to NT1 for large batches, `launch_bounds(...,1)` to `(...,2)`, and sizes the grid from occupancy. Its H200 board speedup rises 1.353→1.492×; verify16 rises 1.154→1.405×. This supports the pipeline/residency hypothesis, but several knobs changed together. [Source diff](prof/moe_v4_v6.diff).\n2. **Large verify dequant/reuse/work balance**: v6 verify32 remains 468.4 us in the profile versus ~160.8 us of practical work per call (~34% efficiency); verify16 is 291.8 us versus ~149.1 us (~51%). Static v4 NT2 spills, and v6 NT1 still trades some spills for occupancy. Routing skew, extra weight passes and instruction issue remain unresolved.\n3. **Cross-CTA phase completion and output reduction**: the fused source polls `C_DONE`, `C_ZERO`, producer mbarriers and global phase barriers. Static instruction chains and source waits establish dependencies, not their actual cost. At bs1, v6 profile is 33.0 us against stock’s 75.4 us, so removing glue was valuable; high-token work now dominates.')
 elif p=='muse_decode_layer':
  w('Top three bottlenecks:')
  w('\n1. **Marlin still dominates verify**: our verify32 profile totals 51.375 ms: experts 35.379 ms (68.9%), dense 5.644 (11.0%), attention 7.406 (14.4%), visible glue 2.518 (4.9%), omitted 0.430. On the sliding shape alone Marlin is 598.6/764.3 us (78.3%). The isolated MoE win has not landed here.\n2. **Dense stream and dependent boundaries at low concurrency**: verify1 is experts 5.823 ms (45.8%), dense 3.618 (28.4%), attention 1.526 (12.0%), glue 1.708 (13.4%). Four split-K GEMVs are already fused with consumer reductions and PDL; compare the existing 1.382× verify1 layer gain before adding a second reduction kernel.\n3. **Full attention at high batch**: our full verify32 FA3 main is 376.4 us of 1,013.6 us, with Marlin 490.2 us. The layer already chooses FA3 on Hopper. A blanket v2 attention substitution would reverse that sound choice. Source dispatch, profile rows and paired verify32 gain of only 1.042× establish the priorities.')
 else:
  w('Top three bottlenecks:')
  w('\n1. **Expert GEMMs**: first-turn profile experts are 1.003 s (44.6%) versus stock’s ~2.05 s; turn1 experts are 33.423 ms (56.1%). Gate/up and down both matter: turn1 sliding MODE1/0 cost 344.3/177.7 us; first full chunk 3,479.5/1,553.8 us. Larger-tile WGMMA helped, but not equally across loads.\n2. **Replicated dense prefill**: first-turn dense is 757.705 ms (33.7%); a 16K chunk’s three grouped cuBLAS calls plus other GEMMs consume ~4.08 ms. Pooler/unpooler account for ~2.2 of 3.1 TFLOPs per chunk by `docs/serving/serving_levers.md`’s source audit. Their row split targets redundant arithmetic, not an assumed launch bottleneck.\n3. **Long-context FA3 plus boundaries**: last first-turn full chunk FA3 is 8.020 ms (43.2%) of 18.571 ms; two boundaries add 0.850 ms. Across the full first turn attention is 263.533 ms (11.7%), glue 220.571 ms (9.8%). v2 wins first-turn 1.564× versus v3’s 1.468×, while v3 wins later turns: a shape-policy hybrid is plausible, but requires its own local and H200 validation.')
 w()
 if p in ('muse_decode_layer','muse_prefill_layer'):
  w('| Load | Experts profile ms H200 ours / stock | Attention profile ms H200 ours / stock | Dense profile ms H200 ours / stock | Glue + omitted profile ms H200 ours / stock |')
  w('|---|---:|---:|---:|---:|')
  for l,t in o['loads'].items():
   z=s['loads'][l];oc=t['components'];sc=z['components']
   def pair(k):return f"{oc.get(k,{}).get('us',0)/1000:.3f} / {sc.get(k,{}).get('us',0)/1000:.3f}"
   def glue(c):return sum(c.get(k,{}).get('us',0) for k in ('glue','unattributed'))/1000
   w(f"| {l} | {pair('moe')} | {pair('attention')} | {pair('dense')} | {glue(oc):.3f} / {glue(sc):.3f} |")
  w()
w('## Serving: verify step critical path and prefill')
w()
w('The target verify length is eight tokens per request: `srt/speculative/dflash_worker_v2.py`, `forward_batch_generation`, `draft_token_num=int(self.block_size)`; [serving audit](SERVING.md). The kernel boards’ q_len1 decode loads do not run in this target path. c=1/4/16/32 means 8/32/128/256 target tokens. c=4 is unscored in the isolated layer problem and is interpolated, not measured.')
w()
w('`sim.serving` serializes the 62 target layers, the drafter, target LM head and 134 all-reduces. Actual overlap and the scheduler/CPU critical path are not measured on our hardware. Layer totals use the mean of all paired SGLang receipts; the component split below uses calibration profile shares, normalized to those totals. Thus **every cell in this table is MODEL**, grounded in H200 measurements; no cell is a measured full-node serving time. The two BF16 expert-layer byte surcharges are included under MoE. Omitted profile time remains separate.')
w()
w('| c | MoE ms MODEL | Attention ms MODEL | Dense ms MODEL | Visible glue ms MODEL | Unattributed ms MODEL | All-reduce ms MODEL | Drafter / LM head ms MODEL | CPU ms assumed | Total ms MODEL |')
w('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
for c,v in a['serving_components'].items():
 w(f"| {c} | {v['moe']/1000:.3f} | {v['attention']/1000:.3f} | {v['dense']/1000:.3f} | {v['glue']/1000:.3f} | {v['unattributed']/1000:.3f} | {v['allreduce']/1000:.3f} | {v['drafter']/1000:.3f} / {v['lm_head']/1000:.3f} | 0, unknown | {sum(v.values())/1000:.3f} |")
w()
w('MoE is the largest identified step component at every rung. At c1 dense + visible/omitted glue are also substantial; at c16/32 the Marlin expert work dominates. Full attention is a local bottleneck on a full layer, but only 16 of 62 layers are full. Removing the modeled LM head or sharding the drafter is lower leverage than changing experts. Unattributed c1 time is too large to assign confidently to glue or GEMMs from the truncated report.')
w()
w('| Prefill load | MoE ms MODEL | Attention ms MODEL | Dense ms MODEL | Visible glue ms MODEL | Unattributed ms MODEL | Total ms MODEL |')
w('|---|---:|---:|---:|---:|---:|---:|')
for l,v in a['prefill_components'].items():
 w(f"| {l} | {v['moe']/1000:.1f} | {v['attention']/1000:.1f} | {v['dense']/1000:.1f} | {v['glue']/1000:.1f} | {v['unattributed']/1000:.1f} | {sum(v.values())/1000:.1f} |")
w()
w('These are target-layer prefill loads, not end-to-end TTFT: prefill all-reduces, CPU/scheduler and extra BF16-layer effects are absent from `sim.serving.prefill_load_us`. Prefill LM-head work and drafter initialization are also outside this model; their costs are unmeasured. “turn 1” is a ~753-token later turn at 80K; “first turn” is the weighted initial chunk sequence. The distinction matters. The model’s 13-turn conversation spends ~25/48/72/78% of stock wall time on prefills at c1/4/16/32. Hidden prompt lengths, prefix caching, chunked scheduling, TTFT and overlap can change this ranking.')
w()
w('### How the planned ports move score')
w()
w('All results below are **MODEL**, `tau=4.16`, 13 turns × 220 output tokens, stock prefill ~4.21 s per conversation. Ratios from isolated H200 kernels are projected onto the corresponding component. Expert ports exclude the two BF16 layers and their modeled byte surcharge; whole-layer ports conservatively leave those two layers fully stock. Prefill savings are scaled by 60/62 for the same reason. Routing distributions and integration differ, so these are sensitivities, not serving predictions or measurements. Whole-layer and component ports overlap and must **not** be added. The score interpolates/extrapolates throughput at equal stock interactivity; the second score caps concurrency at 32. Exact inputs and arithmetic: [helper](prof/analyze.py), [output](prof/analysis.json).')
w()
w('| Port / sensitivity | Step saving c1 / c4 / c16 / c32 ms MODEL | Prefill saving / conversation MODEL | Score × MODEL extrapolated / capped |')
w('|---|---:|---:|---:|')
for l,v in a['port_sensitivity'].items():
 if l.startswith('CPU'):continue
 saves=' / '.join(f'{x/1000:+.3f}' for x in v.get('savings_us',{}).values()) or '0'
 ps=v.get('prefill_saved_us',0)/1e6
 w(f"| {l} | {saves} | {ps:.3f} s | {v['score']:.3f} / {v['capped']:.3f} |")
w()
w('The measured-quality port set (whole decode v3 + whole prefill v3) models about **1.44×**, well below the practical-rate 3.10× ceiling. That gap is work still to do, not a forecast that current files deliver 3×. A full v3 layer port is itself substantial integration: preserving collectives, cache layout, routing and the two BF16 layers is required. `sim.serving`’s rerun differs slightly from `CEILING.md` because more receipts have arrived; raw rerun outputs are [serving](prof/serving.txt), [problem ceilings](prof/sim.txt), [calibration](prof/calibrate.txt).')
w()
w('Scheduler/CPU is **unknown**, not measured zero. With this model, adding 100 us unhidden CPU per step changes score to ~0.997× stock; adding 1 ms changes it to ~0.968×. These sensitivities do not bound TTFT, graph misses, batching or CPU stalls in a real server. Collect an end-to-end nsys trace in the first authorized node run; local sm120 or a device-only profile cannot establish this part of the path.')
w()
w('## Static Hopper code audit')
w()
w('No H200 execution was attempted. CUDA 13.0 `nvcc -cubin -arch=sm_90a -O3 -std=c++20 --expt-relaxed-constexpr -lineinfo -Xptxas=-v` compiled copies of the **unchanged device functions**, explicitly instantiating the templates; only torch host wrappers were removed. [Extractor](prof/compile_static.py), [source hashes](prof/sass/manifest.json), [MoE compiler log](prof/sass/moe.compile.log), [attention compiler log](prof/sass/attention.compile.log), [projection compiler log](prof/sass/proj.compile.log). `cuobjdump` supplies resources/SASS and `nvdisasm --print-line-info` supplies source-correlated MoE SASS. MoE uses the TMA-engine one-dimensional bulk-copy instruction (`UBLKCP.S.G`), rather than a multidimensional tensor-map load. These match source templates, not a byte-identical recovered judge cubin.')
w()
w('Decode’s Triton kernels were compiled inside `tools/img` with Triton 3.8, `GPUTarget(cuda,90,32)` emitting `.target sm_90a`, PDL enabled, aligned CUDA pointers/strides specialized as the real JIT does, and reachable token/split/chunk/warp choices. Generic unaligned compilation produces scalar loads and would give a false diagnosis, so those outputs were replaced. [Driver](prof/sass/compile_decode.py), [metadata](prof/sass/decode_compile.json), per-kernel `.ptxas.log` and `.sass.txt` under `prof/sass/`. Reassembling the PTX with ptxas confirms registers/spills. FA3 and Marlin library cubins are not rebuilt here; their role is measured in the H200 profiles.')
w()
w('| Kernel / specialization | Registers STATIC | Stack; spill stores / loads bytes STATIC | Main load path STATIC | Shared/pipeline STATIC |')
w('|---|---:|---:|---|---|')
for row in [
('MoE v4 NT1 (≤64 tokens)','123','0; 0 / 0','128-bit LDG for activation/output; UBLKCP.S.G for packed weights','5 × 36,864-B bulk stages; mbarriers; one resident CTA intended'),
('MoE v4 NT2 (verify16/32)','168','56; 96 / 136','same bulk path; local spill loads/stores','~208/213 KB dynamic at M128/256, plus compiler static allocation'),
('MoE v6 NT1 (current best, all loads)','96','56; 96 / 128','bulk weight path; local spills present','2 × 36,864-B stages; ~94 KB dynamic at M256; two-CTA residency target'),
('Attention v2 L1,S4','110','32; 0 / 0','LDGSTS.E.BYPASS.128 cp.async; scalar/64-bit metadata','~74 KB dynamic, cluster DSM reduction, no TMA bulk'),
('Attention v2 L8,S3','230','32; 0 / 0','same 128-bit cp.async; more MMA/softmax state','~88 KB dynamic, cluster DSM/global-counter waits'),
('Proj skinny all 13 reachable templates','64–236','0; 0 / 0 for all','LDG.E.NA.LTC256B.128.CONSTANT weights; 128-bit activation LDG','shared partials + cluster distributed shared reduction; no TMA'),
('Proj skinny router NT1 / NT8','112 / 220','0; 0 / 0','128-bit loads, hi+lo MMA','cluster split-K; high NT8 live state'),
('Decode GEMV qkv8 / pool-router8 / unpool32','40 / 40 / 48','0; 0 / 0','40 / 40 / 48 static LDGSTS.E.BYPASS.128 sites','80 / 80 / 96 KiB dynamic; async staging, warp-group MMA; no TMA'),
('Decode QK prep, S8, RoPE','64','32; 0 / 0','33 static 128-bit LDG sites plus narrower metadata','4 KiB dynamic; RoPE/libdevice stack, not allocator spills'),
('Decode boundary C1 / C4 / C16','98 / 96 / 32','0; 0 / 0','128-bit data loads; acquire scalar counters','16 KiB / 4 KiB / 16 B dynamic for C1/C4/C16; global atomics/barriers for C>1'),
('Decode route C4 / post-expert','40 / 115','0; 0 / 0','128-bit main data loads; scalar/64-bit indices','32-B dynamic; top8 shuffle/reduction chains')]:w('| '+' | '.join(row)+' |')
w()
w('Shared sizes above describe source/metadata dynamic allocations, not just cuobjdump’s ABI/static allocation (which includes reserved bytes). MoE’s runtime chooses stages from the device limit; v4 five-stage sizes make a second CTA infeasible even though its register count alone can suggest otherwise. v6’s unused NT2 template compiles with 736/556 spill bytes, but it is never dispatched: reporting that as its hot path would be wrong. QK/attention stack frames have zero ptxas register spills; `LDL`/`STL` alone are not proof of spilling.')
w()
w('**Dequant per weight (STATIC):** v4 `dq8` maps 16 NVFP4 codes to eight BF16 pairs using 7 funnel shifts + 9 logic instructions (one merge plus eight masks), followed by 8 BF16x2 scale multiplies: **24/16 = 1.5 instructions per weight**, excluding loads, scale construction, address math and tensor instructions. The compiled body has 112 `SHF.R.W.U32` and 128 `HMUL2.BF16_V2` sites, consistent with 16 unrolled dq8 instances ×16 weights and 128 pair multiplies. This is not a dynamically weighted issue count. Projection and decode dense kernels have no weight dequant. Attention has FP8 conversion, not NVFP4 weight dequant; decode layer’s expert dequant lives in Marlin and is not statically attributed from these copies.')
w()
w('**Potential stalls (hypotheses):** MoE’s producer `SYNCS.PHASECHK` waits, global phase counters and runtime integer divisions in `decode_unit`; attention’s cluster finalization, softmax shuffle chains (verify has 324 butterfly-shuffle sites versus decode’s 100) and high live-register state; projection’s shared/cluster reductions and low row-CTA counts; decode’s split-K consumers, PDL wait and route top8 reductions. Presence establishes synchronization/instruction dependencies. Without Hopper execution counters it does not establish wait cycles, occupancy achieved, issue utilization or HBM saturation.')
w()
w('## Experiment-loop bottlenecks')
w()
w('Queue = `started_at - created_at`; runtime = `finished_at - started_at`, from completed unique receipts with both timestamps. Duplicates are excluded. These are H200 service wall-clock measurements, not GPU kernel times. The sample includes calibration and failed checks; pending jobs are right-censored and absent. [Per-job arithmetic](prof/analysis.json).')
w()
w('| Problem | Completed unique receipts | Queue min / median / max minutes MEASURED | Runtime median / max minutes MEASURED | Latest observed checks used / cap; scored used / cap |')
w('|---|---:|---:|---:|---|')
for v in a['process']:
 lim=v['latest_limits'];w(f"| {v['problem']} | {v['completed']} | {v['queue_min']:.1f} / {v['queue_median']:.1f} / {v['queue_max']:.1f} | {v['run_median']:.1f} / {v['run_max']:.1f} | {lim['checks']['used']}/{lim['checks']['per_day']}; {lim['scored']['used']}/{lim['scored']['per_day']} (`{v['latest_limit_job']}`) |")
w()
w('External caps remain 20 checks + 10 scored runs per problem/day, reset 00:00 UTC = 09:00 KST; serving 3 checks + 1 scored/day ([README](README.md), [serving rules](serving/RULES.md)). Receipt limit snapshots can count accepted work not yet collected, so “remaining” is not a live quota claim. Queue delay dominates latency to evidence; compile/runtime optimization alone cannot remove that delay. Best projection diagnostics consumed several checks without a whole-candidate win, making per-check information yield important.')
w()
w('### Local versus H200 ranking')
w()
w('Only MoE has a useful overlapping timed set in the inspected artifacts. The following are **filename matches**, not hash-verified local/remote pairs: local logs contain no source hash. Local timing uses a different GPU/routing sample and some Humming kernels dispatch a different implementation by architecture. No causal architecture comparison follows from these data. Hash-bearing local JSON overlap was searched; no matched pair with completed timings was found. Prefill local JSON is tests-only; decode uses a different attention path locally; recent projection local candidates have no completed matching H200 receipt in this snapshot.')
w()
w('| MoE file | Local sm120 speedup MEASURED | H200 speedup MEASURED | Evidence |')
w('|---|---:|---:|---|')
for v in json.loads((R/'prof/local_rank.json').read_text()):w(f"| {Path(v['file']).name} | {v['local_speedup']:.3f}× | {v['h200_speedup']:.3f}× | {link(v['local_log'])}; `{v['h200_job']}` |")
w()
w('The six filename matches agree on **9/15 pairwise orderings (60%; Kendall τ=0.20)**. v2 looks 1.227× locally but is 0.890× on H200; v6 looks 1.221× locally yet is best on H200 at 1.492×. v4→v6 reverses locally versus remotely. Some gaps are within run noise, and the set is tiny/non-random. There is no defensible universal rank-correction factor. Use local timing to screen gross regressions and validate correctness/capture; keep several plausible Hopper designs for remote discrimination.')
w()
w('## Implementation options')
w()
w('Expected gains below are explicitly hypotheses or model sensitivities except the quoted H200 comparisons. Options overlap. The lane does not implement or take ownership of these kernels.')
w()
w('| Priority / bottleneck | Concrete alternative | Expected gain and evidence | What discriminates it |')
w('|---|---|---|---|')
w('| 1. Verify layer experts | Port the fused best MoE into the existing v3 layer; preserve router, BF16 layers and collective boundaries. Compare NT1/two-stage resident scheduling to NT2 and tune phase work ordering. | Isolated H200 v6 verify 1.635/1.405/1.211×; projecting those onto layer experts gives the score sensitivity above (~1.098× alone). v4→v6 overall +10.3% measured supports residency tuning. No claim that isolated routing gains transfer exactly. | End-to-end verify H200 layer receipts at all sizes; counter evidence for bandwidth vs spills/phase waits. |')
w('| 2. Prefill redundant dense math | Row-split pooler/unpooler around reduce-scatter/all-gather; choose token threshold from sizes/communication, retain exact BF16 weights. | SERVING source/math estimate ~2 ms/layer/16K chunk, ~0.5 s/first turn; score sensitivity ~1.094× alone. Dense is 33.7% of our first-turn profile. Savings after collectives are unmeasured. | TP8 H200 collectives and output agreement; first-turn layer and whole-node TTFT. |')
w('| 3. Prefill expert shape policy | Combine v3 WGMMA later-turn path with the faster measured v2 first-turn path; then vary grouped tile layout, producer stages and gate/up SwiGLU epilogue. | Existing first-turn ratios 1.564× v2 vs 1.468× v3 imply ~6.1% less first-turn time, if a hybrid reproduces both; v3’s overall gain comes from later-turn wins. Practical expert/math rates remain targets. | One hybrid receipt; sweep stage/token tile and measure per-phase profiles rather than assuming all shapes improve. |')
w('| 4. Low-c target layer launch/glue | Fuse QK norm+store+query cast; port MoE finalize; consider a persistent layer kernel with PDL boundaries where weights/collectives permit it. PDL already exists in v3, so merely enabling it is not new. | SERVING finalize estimate 0.7–1.5 ms at c1; score ~1.014–1.029× alone. sim practical 8→1 target kernels saves 62×14 us = 0.868 ms/step, score ~3.10→3.17× on the already optimized model. Benefits are not additive with prior glue fusion. | nsys node timeline, graph gaps/PDL overlap, correctness with real all-reduces; avoid residency deadlocks in global barriers. |')
w('| 5. Attention verify tile/layout | Keep winning FA3 verify dispatch until a GQA verify tile reuses K/V across all eight queries with lower live state. Compare warp-specialized/TMA staging against current cp.async and cluster split counts. | Current v2 all-verify port models a regression (~0.991× score). Recovering its 0.08/0.36/1.08/1.09 ms step penalties would only return to stock. Practical attention model has further headroom, but no measured replacement gain. | H200 verify profile at page size1, soft/coherent data and every key; register pressure, page-gather efficiency and split finalization. |')
w('| 6. Dense skinny split-K | Tune NT/TR/KS per operation; use lossless prepacked layouts and cluster partial reduction; gate router high-M to the faster measured design. Reduce live accumulators before adding prefetch depth. | Router verify32 28.7 us vs ~2.7 us floor is a clear gap; whole sweep loses high-token loads. Target 10 us off router ×62 ≈0.62 ms/step is a hypothesis, not a measured gain. NT8 router 220 registers; no allocator spills in static code. | Standalone paired router H200 timings; held-out nearby M; cluster reduction cost and width-preserving loads. |')
w('| 7. Judge queue and check yield | Bundle N candidate variants into one **diagnostic timed callable**, each with distinct kernel names, all work included in the timed graph; return a correctness-checked output and inspect per-kernel profiles. Keep N small enough for compilation/test limits. | N=4 can reduce four sequential queue waits to one: roughly 3×35–73 = 105–219 minutes saved in serial turnaround, and 4→1 checks. This is arithmetic on observed waits, not queue-throughput prediction. Combined timing is not a competitive speedup. | Verify each variant independently locally; preserve cold-cache caveat (later variants are warm), correctness and source rules. No input-dependent timing/work in prepare, no harness changes or timing patches. |')
w('| 8. Communication / drafter | Fuse finalize into all-reduce only after its producer writes the real workspace; token-gate dense sharding; shard drafter replicated FC with all-gather. | MODEL all-reduce 0.80→2.77 ms c1→32; FC estimate 0.15 ms/step gives only ~1.004× score. Unconditional pooler sharding loses ~0.45 ms at c32; gate ≤128 tokens pending measurements. | Actual TP8 NVSwitch latency, multicast availability, overlap and final output agreement. |')
w()
w('## What is not determined')
w()
w('No H200 counters: achieved HBM/L2 bandwidth or hit rate; memory transaction efficiency for page1 gathers; tensor/integer issue utilization; actual occupancy/eligible warps; spill traffic cost; barrier/scoreboard wait cycles; producer/consumer imbalance; cache eviction and true PDL overlap. Static SASS cannot supply those. H200 traces or richer retained profiles are also needed for omitted kernel names and exclusive GPU critical path; counters alone do not restore truncated names. Full-node serving is additionally required for all-reduce/multicast, CPU/scheduler, prefill interference, acceptance/tau and the hidden workload/score curve. There is no measured serving score in this report.')
w()
w('Verified here: stdlib-only execution over all five problems and serving-only filters; exact selection from receipts; profile row accounting including omitted mass; model reruns; public upstream file equality; sm90a compile success/resources/SASS for the requested kernels and the new MoE best. Not verified: Hopper execution of these local cubins, current kernel integration, new port correctness/performance, or node throughput. The reports preserve the distinction between what was observed and what was modeled.')
import re
md='\n'.join(out)+'\n'
md=md.replace('](SERVING.md)','](../serving/serving_levers.md)')
md=re.sub(r'\]\((?!https?:|#|\.\./|serving_levers\.md)','](../../',md)
(R/'docs/reference/profile.md').write_text(md)
