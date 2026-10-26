"""Report the matched cohort without mixing historical nodes or selecting rounds."""
import json
import math
from pathlib import Path

D = Path(__file__).resolve().parent
read = lambda name: json.loads((D / name).read_text())
r = read('results.json')
assert r['measured'] and not r['invalid'] and r['source_audit']['verified']
ms = (1, 8, 16, 32)
summary = {(v['M'], v['arm']): v for v in r['summary']}
contrasts = {(v['M'], v['term']): v for v in r['contrasts']}
layer = {int(k): v for k, v in r['modeled_layer_us'].items()}

def ceiling(deltas):
    return math.exp(sum(-math.log1p(-max(0, d) / layer[m]) for m, d in deltas.items()) / 6)

excess = {m: summary[m, 'BASE']['mean'] - summary[m, 'STREAM']['mean'] for m in ms}
act = {m: contrasts[m, 'activation_transport_sink']['mean'] for m in ms if m != 1}
lo, hi = 0., min(act.values())
for _ in range(80):
    mid = (lo + hi) / 2
    if ceiling({m: max(0., v - mid) for m, v in act.items()}) > 1.01:
        lo = mid
    else:
        hi = mid
largest = [{'M': m, 'term': 'epilogue_NOWAIT' if m == 1 else 'MMA_activation'} for m in ms]
c = dict(stream_excess_us=excess, stream_ceiling_R=ceiling(excess),
         one_third_stream_excess_R=ceiling({m: d / 3 for m, d in excess.items()}),
         MMA_activation_M8_32_ceiling_R=ceiling({m: contrasts[m, 'MMA_activation']['mean'] for m in ms if m != 1}),
         activation_reference_M8_32_ceiling_R=ceiling(act),
         activation_reference_common_cost_budget_for1pct_us=lo, largest=largest,
         assumption='Planning only; full stage delta credited to layer; 46/16 S18->S19 transported load times; not held-out G0')
(D / 'ceiling.json').write_text(json.dumps(c, indent=2) + '\n')

def cell(v):
    return f"{v['mean']:.2f} [{v['CI95'][0]:.2f}, {v['CI95'][1]:.2f}]"

lines = ['# Matched S19 residual assignment', '',
         f"Receipt **{r['job']}**, prefill-host check. P3a remains **active**. No new point winner goes to P4.", '',
         'All times are us: mean [95% CI] across eight seed means, each using all four symmetric rounds. '
         'Absolute intervals include route/expert variation; paired contrasts remove much of it. '
         'This is an exploratory stage cohort, not held-out whole-layer G0 confirmation.', '',
         '| Arm | M1 | M8 | M16 | M32 |', '|---|---:|---:|---:|---:|']
for arm in r['seed_level_record']['arm_names']:
    lines.append('| ' + arm + ' | ' + ' | '.join(cell(summary[m, arm]) for m in ms) + ' |')
lines += ['', '## Calibrated contrasts', '',
          'Same seed, same translation unit, 264 CTAs, 256 threads, depth 2 and shared reservation. '
          'All 60 runtime resource rows match offline sm90a counts: two resident CTAs, zero spills. '
          'Raw/decode/MMA sink schedules are matched to prevent dead-code elimination. '
          'These are conditional schedule contrasts; they do not identify independent physical pipe latencies.', '',
          '| Contrast | M1 | M8 | M16 | M32 |', '|---|---:|---:|---:|---:|']
for term, label in [('load_issue_repeats', 'L0−STREAM: raw load-side schedule/repeats'),
                    ('live_sink_raw', 'L0S−L0: raw sink calibration'),
                    ('decode_scale', 'L1−L0S: decode/scale'),
                    ('MMA_activation', 'L2−L1: MMA + activation transport'),
                    ('epilogue_NOWAIT', 'L3−L2: epilogue NOWAIT'),
                    ('dependency', 'BSINK−L3: dependency with live sink'),
                    ('live_sink_normal', 'BSINK−BASE: normal sink calibration'),
                    ('activation_transport_sink', 'ACT−LOOP: activation + operand sinks reference'),
                    ('tile_schedule', 'LOOP−META: tile schedule reference')]:
    lines.append('| ' + label + ' | ' + ' | '.join(cell(contrasts[m, term]) for m in ms) + ' |')
lines += ['', 'Checked per seed, not just at means:', '',
          '`BASE = STREAM + (L0−STREAM) + (L0S−L0) + (L1−L0S) + (L2−L1) + (L3−L2) + (BSINK−L3) − (BSINK−BASE)`', '',
          'This accounts for the observed residual without importing an old stream node or clock. '
          'Sink tax is material (2.14/5.52/7.42/13.42 us in the normal kernel). '
          'Decode is not shown to saturate an integer pipe by this experiment. '
          'MATH retains timed on-chip operand initialization, small GS/norm global accesses, output and epilogue; '
          'it removes global raw-weight copies, not every global instruction. '
          'ACT−LOOP retains live operand sinks and is a reference for a layout oracle, not exclusive activation memory latency. '
          'The unresolved M8 L0−STREAM interval crosses zero; it is not negative physical work.', '',
          '## Work and finish observations', '',
          'The 8,448 CTA observations retain joint (tiles, experts) histograms; every seed/M retains exact '
          'entry/W2-end/finish endpoint order statistics. '
          'A separate, prospectively fixed seed-0 phase in the same run retains per-SM CTA/tile/expert counts and last finish, '
          'plus joint work/finish histograms at 2.048 us bucket precision. '
          'Last-CTA endpoint order statistics themselves retain ns precision. '
          'HIST tax is measured below; tails overlap the ladder terms and cannot be added to them.', '',
          '| Observation | M1 | M8 | M16 | M32 |', '|---|---:|---:|---:|---:|']
for term, label in [('entry_spread_us', 'CTA entry spread'), ('W2_tail_p50_to_last_us', 'W2 end p50→last'),
                    ('finish_tail_p50_to_last_us', 'Kernel finish p50→last')]:
    lines.append('| ' + label + ' | ' + ' | '.join(cell(next(v for v in r['tail'] if v['M'] == m)[term]) for m in ms) + ' |')
for key, label in [('observer', 'HIST/BASE time'), ('AA', 'A:A time')]:
    vals = [next(v for v in r['tail'] if v['M'] == m)[key] for m in ms]
    lines.append('| ' + label + ' | ' + ' | '.join(f"{v['R']:.5f} [{v['CI95'][0]:.5f}, {v['CI95'][1]:.5f}]" for v in vals) + ' |')
lines += ['', 'Tile counts vary with actual touched experts: across all seeds the per-CTA ranges are '
          '4–12 / 13–27 / 16–30 / 22–42; experts per CTA are 1 / 1–2 / 1–2 / 1–2. '
          'The arrays preserve each cohort member; comparing extrema across different seeds is not an imbalance measurement. '
          'The within-member W2 tail is measured, but moving work could also move the readiness boundary.', '',
          '## Largest terms and six-load ceilings', '',
          'Planning model only: each stage saving is credited in full to every sliding/full layer of its affected load; '
          'the other loads are unchanged. Layer times are 84.82/126.97/176.09/221.25 us, transported from '
          'official S18 weighted times by the S19/S18 paired load ratios. '
          '`R = exp(sum(-log(1−delta_M/T_load))/6)`. Mandatory MMA cannot simply be removed; '
          'these ceilings rank exposed work and do not predict a realizable win.', '',
          '| M | Largest added stage term | us [95% CI] | Six-load R if removed |', '|---|---|---:|---:|']
for v in largest:
    m, term = v['M'], v['term']
    cv = next(v for v in r['ceilings'] if v['M'] == m and v['term'] == term)
    lines.append(f"| {m} | {term} | {cell(contrasts[m, term])} | {cv['modeled_R']:.5f} |")
lines += ['', f"The M8–32 MMA/activation increments together give modeled R **{c['MMA_activation_M8_32_ceiling_R']:.5f}**. "
          f"M32's 15.70 us load-side gap alone gives R **1.01234**. "
          f"BASE−STREAM is 11.99/15.65/22.96/47.10 us: free removal gives R **{c['stream_ceiling_R']:.5f}**, "
          f"one third gives **{c['one_third_stream_excess_R']:.5f}**. These terms keep the lever active.", '',
          '## Built decision: EARLY slot release', '',
          'The initial matched L2−L1 signal selected a compute/load overlap change before the corrected final cohort: '
          'after the final raw vector and scale reads, release the slot and issue the next TMA copy before the last '
          'decode/MMA group and partial-store barrier. Explicit volatile g7 shared loads and a warp sync keep '
          'the last operands safe in registers. Eight math warps, two slots, exact math/bytes, '
          'readiness ordering and two-CTA residency remain. This differs from adding a deeper slot or a producer warp.', '',
          'EARLY uses 128 registers, zero stack/spills and 84,176/84,496/85,136 shared bytes at M8/16/32. '
          '135 native TASK checks cover nearby M9/M24, real/all/empty experts, start offsets 0/96/224 and changed inputs '
          'under a reused graph. Normal and forced integration screens each pass 13 TASK checks/12 shapes with the '
          'changed MoE leaf native; only unrelated Hopper branches use established stand-ins. '
          'Remote TASK controls pass before and after the cohort.', '',
          '| M | EARLY/S19 time [95% CI] | paired log SE / A:A log SE | Decision |', '|---|---:|---:|---|']
for v in r['candidate']:
    lines.append(f"| {v['M']} | {v['ratio']:.5f} [{v['CI95'][0]:.5f}, {v['CI95'][1]:.5f}] | "
                 f"{v['log_SE']:.6f} / {v['AA_log_SE']:.6f} | " + ('M1 control alias, excluded' if v['control_alias'] else 'Measured loss; no nomination') + ' |')
lines += ['', 'Candidate ratio intervals use promotion.py on the eight paired seed means with max(paired, A:A) log SE. '
          'The direct slot-release placement loses; it does not close activation layout, MMA scheduling or M32 load issue. '
          'EARLY_PRIVATE is an offline-only register-scope prototype, not a measured loss or nomination. '
          'Exact tested patch and candidate SHA are in early/manifest.json; P4_HANDOFF.json exports no new winner. '
          'Existing point_nominations.json rows are preserved. No held-out whole-layer R was spent on a stage point loss.', '',
          '## Active next bound', '',
          'The next cheapest ranking test is an on-chip/pack-free activation-operand layout oracle, paired with BASE '
          'and ACT/LOOP controls, then a correct timed coalesced BF16 operand bank only if the oracle is material. '
          'S19 dynamically loads Bf operands in the math warps; an expert/token-block register-order bank could '
          'remove gathers/address work while preserving BF16. For W13, packing belongs in the timed route/producer; '
          'for W2, write the bank directly from W13. No activations move into prepare.', '',
          'One padded 8-token×4096 BF16 block is 64 KiB; at 32 experts×4 blocks a bank reserves at most 8 MiB '
          '(16 MiB for both phases). These are allocation ceilings, not a claim that every padded row must be copied. '
          'The design must keep the existing two-CTA shared/register budget; no HBM savings are assumed. '
          f"ACT−LOOP's M8–32 reference ceiling is R **{c['activation_reference_M8_32_ceiling_R']:.5f}**. "
          f"If all that reference were removable, a common added packing cost must stay below **{lo:.2f} us** "
          'per affected load for >1% modeled six-load R. The reference includes sink cost, so the pack-free oracle '
          'must first establish a real exposed benefit. This is sized and unimplemented, not a nominated kernel. '
          'M32 raw load-side scheduling is a second >1% opening. Dependency/DSM forms remain scoped closed or '
          'unadmitted by their earlier costs; they do not justify parking these terms.', '',
          '## Receipt and transport audit', '',
          f"Final {r['job']}: wrapper {r['source_audit']['wrapper_sha256']}, both checksummed packets, embedded cell "
          'identities, all 2,048 round times, 96 NVML snapshots, 60 resource rows and work/finish records verified. '
          'All correctness-bearing BASE/AA/BSINK/HIST/EARLY controls pass. Timing-only cuts return ignored outputs. '
          'The host passed=false is the declared lab-only prepare stop; it is not a failed native control.', '',
          'c43111f584854b05 is unmeasured: schema/output transport failed and replaced the timing archive with '
          'an explicit oversize packet. da1fb2eeb0d3443d has valid controls, but its scalar STREAM and per-tile ACT sink '
          'measure those costlier recipes, not the requested pure-byte/activation oracles. initial_* snapshots retain '
          'that evidence; no row above mixes their times with the corrected cohort. Local final exact payload packets '
          'are 7,480/2,143 bytes versus declarations 8,600/2,800. Sources and receipts are retained in runs/.', '']
if (D.parent / 'operand/queue.json').exists():
    follow = json.loads((D.parent / 'operand/queue.json').read_text())
    lines += ['## Active operand follow-up', '',
              f"Matched operand/issue batch queued as sender **{follow['sender_pid']}**, wrapper "
              f"**{follow['wrapper_sha256'][:16]}**, for the10-27 09:00KST quota reset. "
              'New H200 us/CIs are pending. [Oracle arms, resource budgets and production-graph decision]'
              '(../operand/REPORT.md). No per-call result is treated as a transferred S21 score.', '']
(D / 'REPORT.md').write_text('\n'.join(lines))
print('reported', r['job'], c)
