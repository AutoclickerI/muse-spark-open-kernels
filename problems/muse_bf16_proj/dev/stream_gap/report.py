"""Readable per-shape decomposition, generated only from retained receipts."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
d=json.loads((H/'decomposition.json').read_text());f=d['stream_fit'];nop=d['nop']
lines=['# P10 projection leader gap','',
'The 1.36125 time ratio is a component-board aggregate (`08cc94050a4e4564`), not a measured ratio between the two teams inside a layer. The leader publishes neither its per-shape times nor its implementation. This report identifies our costs and tests transferable changes; it cannot attribute an exact algorithm to that team.','',
'`2015fc90abf94e14` completes all 30 projection shapes, eight independent activation seeds and A:A. It compares exact P12/S15 control A, a generic exact 12-bit decoder D storing BF16 directly, identical generic FP32 output plus cast F, and the current producer P without the component adapter. M1 in this first batch uses one-hot timing; its random-input confirmation remains a limit.','',
'## Bytes and stream rate','',
f'The checksum-verified cold NC stream has a fitted marginal rate of **{f["marginal_TBs"]:.3f} TB/s** and **{f["intercept_us"]:.3f} us** intercept over 32–128 MiB; maximum end-to-end rate is {f["max_effective_TBs"]:.3f} TB/s. Compare with the earlier judge stream 4.58 TB/s and TASK peak 4.8 TB/s. This is a size regression, not a confidence interval. Receipt: `2015fc90abf94e14`.','',
'B/W is the complete one-pass logical weight representation including tables/escapes/padding. It is **not a hardware DRAM counter**: repeated CTA requests, cache reuse, transactions, activation loads and output/partial writes can change physical traffic. The rates below divide that bank size by the measured current producer time P, so they are useful-throughput rates. The CSV/JSON retain board A, D, cast costs, all intervals, graph node counts and stream-fit residuals.','',
'| M | Role | A B/W | D B/W | P us | Useful TB/s | /4.58 | /4.8 | A / D launches |',
'|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for r in d['rows']:
    lines.append(f'| {r["t"]} | {r["role"]} | {r["A_bytes_per_weight"]:.4f} | {r["D_bytes_per_weight"]:.4f} | {r["P_us"]:.3f} | {r["producer_effective_TBs"]:.3f} | {r["producer_fraction_of_best_stream"]:.1%} | {r["producer_fraction_of_peak"]:.1%} | {r["nodes_A"]} / {r["nodes_D"]} |')
lines += ['',
'The control already codes QKV M1/M32, O M1/M8/M16 and unpool M16/M32. Its coded totals are roughly 1.50–1.57 B/W; router remains two BF16 banks, 4 B/original FP32 weight. Raw remaining BF16 sites are 2 B/W. A blanket extra 25% reduction therefore overstates the available saving.','',
'## Fixed work and output boundaries','',
f'Empty graphs with 0/1/2/4/8/16 no-op nodes take '+', '.join(f'{int(n)}:{v:.3f}' for n,v in nop['raw'])+f' us. The 2–16-node fit is **{nop["marginal_launch_us"]:.3f} us/additional launch**; the first node adds {nop["first_launch_minus_empty_us"]:.3f} us. The 2.876-us stream intercept is consequently not all removable launch time. Receipt: `2015fc90abf94e14`.','',
'Most small component shapes have a producer plus a reduction/cast node; the large router has three nodes. Generic direct BF16 output and the exact hi/lo router reduce those to one. F minus D isolates 1.22–1.79 us for the generic output boundary. Current A minus P is about 1.2–2.1 us at split/cast sites (independent captures, not an additive critical-path measurement). These conversion wins explain part of a component-board gap; the real layer generally passes FP32 partials directly to a fused consumer.','',
'D/A wins at pool M1/8/16/32 are .91118/.75085/.76407/.80414; router wins at M1/8/16/32/128/256 are .59450/.64688/.68029/.74258/.63111/.74565, with per-cell CIs in decomposition.json. O1/8 direct-store wins are .77826/.81828, while the identical generic FP32+cast recipes are .96295/1.01834: much of this apparent gain is a board-output boundary. Generic coded QKV loses at every tested small M (1.039–1.154); unpool loses (1.060–1.249).','',
'The optimistic same-receipt per-shape component selection is 1.10711x. Its joint samples were not retained, so no composition CI exists. It is selection data, not a G0 result.','',
'## Fusion and large-M reuse','',
'`646822ab3d164be8` uses 16 independent random activation/residual seeds, eight cold alternating rounds and A:A, retaining every joint sample. O1 last-CTA fusion combines the projection, residual and two norms in one launch. The current NF O1 + cluster boundary costs 8.102 us; WR4/2/1 fusion costs 12.984/15.971/23.872 us. Ratios [95% paired-log t15 intervals]: **1.602651 [1.588091,1.617346]**, **1.971324 [1.954076,1.988723]**, **2.946596 [2.927237,2.966083]**. A:A is .986591 [.976728,.996554]. This serial finalizer loses despite saving a launch.','',
'The same receipt tests 20 full-M-tile/split-K alternatives at M128/M256 against each current projection with the same route/norm or residual/norm neighbour. Every arm loses: 1.102–1.701 at M128 and 1.521–3.734 at M256. These recipes request each logical weight bank once across their disjoint K splits, but pay a larger accumulator/shared-memory footprint and changed grid. Native builds use 73.8–98.3 KB shared and 90–190 registers, with zero spills. Which resource causes the remaining slowdown is not isolated; footprint and occupancy are source evidence, not a causal timing decomposition. `followup_results.json` retains every interval and A:A sample.','',
'The board pool N4096 and standalone router are not interchangeable with the layer fused pool+hi+lo N4608. P4 must retain fused routing, QKV3/gate concurrency, the FP32 partial ABI and PDL ordering. The already-measured small-M pool dispatch is in S17/S18; it is not a new nomination from this batch.','',
'## Current build and P4','',
'The new build tests exact 12-bit QKV at M8/M16, both full N2304 and split N1280. Six geometries vary WK4/8, S1/2/4 and one/two pipeline stages. It changes neither the decoder precision nor the consumer format. All 24 arms and 16 fresh activation seeds are batched on the projection host; samples include independent A:A and separate TASK one-hot checks. `qkv_admission.json` pins local normal/forced 9 TASK tests + 30 graph shapes, all 24 extra cells, 24 zero-spill sm90a units and 24 native host-dispatch stand-ins.','',
'Read the accepted receipt with `read_structural.py <job>`, then regenerate this report. P4 handoff is `P4_HANDOFF.json`; no layer promotion is claimed without held-out paired six-load R and lower 95% bound >1.','',
'Rebuild: build.py -> build_lab.py (first decomposition); build_o1.py / build_followup.py (structural recipes); build_qkv.py (new batch). The router port is retained with its full MIT notice, so rebuilding no longer depends on a previous agent temporary checkout. Generated candidates, wrappers and local outputs are ignored.']
q=H/'qkv_results.json'
if q.exists():
    r=json.loads(q.read_text());lines += ['',f'QKV dispatch receipt: `{r["job"]}`; outer host passed: {r["outer_host_passed"]}. Ratios are candidate/control time; point winners require P4 held-out confirmation.','',
    '| M | Role | Arm (WK,S,stages) | Ratio [95% CI] | A:A [95% CI] |','|---:|---|---|---|---|']
    for t,role in sorted({(x['t'],x['role']) for x in r['records']}):
        rows=[x for x in r['records'] if (x['t'],x['role'])==(t,role)];aa=next(x for x in rows if x['arm']=='AA')['ratio']
        for row in rows:
            if row['arm'] in ('A','AA'):continue
            v=row['ratio'];lines.append(f'| {t} | {role} | {row["cfg"]} | {v[0]:.6f} [{v[1]:.6f},{v[2]:.6f}] | {aa[0]:.6f} [{aa[1]:.6f},{aa[2]:.6f}] |')
    lines += ['',
    'P4 receives arm4 (WK4/S4/ST1) at M8 full QKV, M16 full QKV and M16 QKV3. The M8 QKV3 win is retained as a diagnostic because that site is inactive in the current layer. patch_qkv.py prunes untimed prepare to those three instances/two banks. Applied to shipped S18 e04635ec it produces f9e4661d; layer_qkv_admission.json records 13 TASK tests and 12 graph shapes in normal and forced local modes, executing every changed site, plus three zero-spill Hopper builds. P4 G0 remains unmeasured; the additional H200 prepare cost on that base is also unmeasured.']
(H/'README.md').write_text('\n'.join(lines)+'\n')
