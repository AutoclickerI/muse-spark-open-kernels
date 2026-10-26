"""Current report and P4 handoff, with pending native results kept explicit."""
import json
from pathlib import Path
D=Path(__file__).resolve().parent
read=lambda name:json.loads((D/name).read_text())
q=read('queue.json');m=read('manifest.json');counts=read('counts.json')['rows'];lab=read('lab_manifest.json');screen=read('rehearsal_validation.json')
out=read('results.json') if (D/'results.json').exists() else None
mechanisms={
 'CACHE1':('Real64KiB BF16 activation bank,two vector planes,one raw TMA slot','Fragment placement/Bf lifetime;real fill/LDS/one-slot costs paid'),
 'LAZY':('Load original BF16 fragments per K-group/tile instead of keeping64 registers live','Bf lifetime/activation scheduling;register-lifetime control for LAZY2'),
 'LAZY2':('LAZY plus two alternating FP32 accumulator chains,final sum','Accumulator dependency latency with freed register capacity'),
 'N1SIMT':('FP32 SIMT matvec only for actual singleton token blocks;HMMA otherwise','Useful-N padding;all K,experts,outputs retained'),
 'EPI4':('Four epilogue warps,scalar W2 channel ownership;remove duplicate norm flush','Epilogue serialization;original reduction/readiness retained')}
lines=['# P3a correct S22 outcome portfolio','',
 'Active. Native H200 results pending.' if out is None else f"Receipt **{out['job']}**; per-arm validity is reported below.",
 f"Frozen base S22 expanded **{lab['base_sha256']}**. Paired sender **{q['sender_pid']}**, wrapper **{q['wrapper_sha256']}**; `--lane P3a`, PLAN rank4,MoE check after09:00KST10-27. Older diagnostic sender3637718/wrappere1710fb3 precedes it. No scored submission.",'',
 '| Candidate | Mechanism | Diagnostic outcome targeted |','|---|---|---|']
for name,(mechanism,target) in mechanisms.items():lines.append(f'| {name} | {mechanism} | {target} |')
lines+=['','| Candidate | Registers M1 / M8+ | Shared B M1 / M8 / M16 / M32 | CTAs/SM | Stack/spills |','|---|---:|---|---:|---|']
for name in m:
 c=[next(v for v in counts if v['arm']==name and v['M']==M) for M in (1,8,16,32)]
 lines.append(f"| {name} | {c[0]['registers']} / {c[1]['registers']} | "+' / '.join(str(v['shared']) for v in c)+' | 2 | 0 |')
lines+=['','| Candidate | Whole six-load speedup R [95%CI] | Per-shape point nominees | Leaves open |','|---|---|---|---|']
for name in m:
 v=out.get('outcomes',{}).get(name,{}) if out else {}
 result=f"{v['R']:.6f} [{v['CI95'][0]:.6f},{v['CI95'][1]:.6f}]" if v.get('measured') else 'unmeasured;await native receipt' if out is None else 'unmeasured;failed/missing native evidence'
 points=','.join(x['label'] for x in v.get('shapes',[]) if x['point_nominee']) or 'none yet' if out is None else ','.join(x['label'] for x in v.get('shapes',[]) if x['point_nominee']) or 'none'
 lines.append(f'| {name} | {result} | {points} | Other schedules/layouts and P4 composition;only this exact recipe can close |')
lines+=['',
 '**Correctness and local scope.** All five exact candidate math bodies pass270 leaf TASK gates each: '
 'M1/8/9/16/24/32,real/all/empty/singleton/dense routes,local_start0/96/224,three changed inputs per captured graph. '
 'All five shipped candidates pass normal and forced Hopper-glue screens:13TASK cases and12captured shapes per mode '
 '(130TASK,120graphs total). All three small-M native launch paths are observed eagerly and captured. '
 'Fifteen exact sm90a translation units compile;20shape/layout occupancy rows retain two CTAs and zero spills. '
 'CACHE1 needs113KB shared memory unavailable on sm120:only its activation-bank address space/LDS is replaced '
 'with preallocated global storage locally;math,one-slot raw TMA,ordering,allocation and native launch remain. '
 'The H200 payload contains the actual shared bank. Other Hopper-stage stand-ins are the existing S22 screen hooks. '
 'Native H200 correctness and performance remain unmeasured until the receipt.', '',
 f"**Transport rehearsal.** The exact packed payload passes {screen['TASK']}TASK+{screen['nearby']}nearby checks on all12graphs, "
 'with an explicit local two-seed/two-round plan and forced stand-ins. This is not a64-seed H200 result. '
 f"Lossless archive round-trip passes; synthetic64×16 packet stress is{screen['synthetic64x16_transport_bytes']}B "
 'within11744B,including four complete retained timing traces with NVML. Synthetic values never enter results. '
 'The codec losslessly packs the existing six-decimal sample integers and regenerates deterministic round orders; '
 'all six loads,seeds and round counts remain. Receipts use checksums and immutable source audits.', '',
 '**Native measurement plan.** ec754e5/S21 protocol:64 disjoint held-out seeds×16 rotated forward/reverse rounds, '
 'arithmetic means of every round,cold-cache flushes,all12whole-production-graph shapes for every arm. '
 'A and A2 are separately captured exact S22 graphs using the common baseline preparation;five candidates give5376TASK+780nearby checks. '
 'Actual per-seed six-load times use46sliding+16full;no off-shape imputation or component-time score transfer. '
 'Every primary six-load seed sample and round count is kept,plus bounded worst4 per-round GPU/enqueue traces '
 'and NVML before/after timing blocks. Targeted eight small-M shape moments retain paired log-SE;four heavy '
 'shape moments are redundant and omitted,but their actual six-load samples remain. Prepare is reported '
 'as shared host preparation plus each arm delta,a conservative bound rather than a cold candidate measurement.', '',
 '**Gate and handoff.** `read.py` audits wrapper,payload,bundle,checker identities and calls `promotion.evaluate`. '
 'Every valid whole-layer shape point win goes to P4 with ratio,SE,seed IDs and exact S22 patch. '
 'A:A sets uncertainty without vetoing individual shapes. P4 composes and confirms on new held-out seeds; '
 'six-load R lower95%bound>1 plus TASK/nearby establish the lab gate. Official TASK and the300s prepare '
 'limit are still required for final submission. No measured winner exists before the native receipt.', '',
 '**Finite follow-through.** `follow.py` waits on receipt/file and sender-exit events,reads the diagnostic, '
 'and removes an arm only if the predeclared matched upper saving bound is nonpositive at every M '
 '(LAZY requires a measured loss at every M). A merely non-largest positive term is retained. '
 'Failed controls/implementations are unmeasured and do not close a recipe. Only the owned unsubmitted '
 'paired waiter can be replaced under the dispatch lock;an accepted job is never cancelled. '
 'If dispatch outruns the diagnostic,all arms stay. The worker reads the one eventual paired receipt, '
 'exports P4_HANDOFF/point_nominations and ends;no automatic scoring or shared-doc writes.', '',
 '**Exact rebuild.** `base_snapshot.json` pins the original packed S22 Git blob,so P4 edits cannot silently '
 'change the base. Run `build.py`,then `count.py` inside tools/img,`check.py`,`run_screens.py`,`build_lab.py` '
 'and `rehearse.py` on GPU0. `manifest.json` pins expanded/shipped/patch/cell hashes. Each patch applies '
 'to the expanded `dev/s22/generated/s22.py`,not the compressed shipment. Only the three small-M CUDA '
 'literals change;all notices and remaining graph code are retained. Native resource checks repeat on H200.', '',
 'One EPI4 M1 local failure was traced to SLOT metadata never initializing expert-wide n/off. Keeping '
 'the actual segment off/ne fixed it;every leaf,normal/forced and packed check reran on the final source. '
 'It is an implementation fix,not an H200 loss. No closed mixed_wgmma/public-port design is reused.', '']
(D/'REPORT.md').write_text('\n'.join(lines))
if out is None:
 hand=dict(base='S22 '+lab['base_sha256'],job=None,rows=[],state=q['state'],sender_pid=q['sender_pid'],wrapper_sha256=q['wrapper_sha256'],candidate_sources=lab['candidate_sources'],whole_production_graph_measured=False,gate='Pending native receipt;no local performance nomination',diagnostic_sender=3637718)
 (D/'P4_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n')
elif not out.get('outcomes'):
 (D/'P4_HANDOFF.json').write_text(json.dumps(dict(base='S22 '+lab['base_sha256'],job=out['job'],rows=[],state='Unmeasured:missing complete native archive',reason=out.get('reason')),indent=2)+'\n')
print('P3A22_REPORT',q['state'])
