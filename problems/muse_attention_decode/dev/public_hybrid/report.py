"""Current measured decisions, complete points in result.json, scoped closures."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];r=json.loads((H/'result.json').read_text());sizing=json.loads((H/'sizing.json').read_text());hand=json.loads((H/'P4_HANDOFF.json').read_text())
f=lambda p:f"{p['r']:.6f}[{p['lo']:.6f},{p['hi']:.6f}]"
best={c:min((p for p in r['points'] if p['cell']==c),key=lambda p:p['r']) for c in r['controls']}
s='''# Post-port attention hybrids

Native **cde88b9136de4093**, proj check, complete224.86s lab: **780TASK**
checks pass,8freshseeds10739123+7919*i, all4752integer-nanosecond samples
retained. All cyclic rotations and exact reverses,2N rounds for N arms;
each arm every exposure position twice. ColdL2/fixed graph subtraction,
eager+poison/repeated graphs, same input per paired arm, no filtering.
Licenses R-20261026-1: operatorstack6d9b7765 MIT, Lee601c6541 Apache2;
full notices retained in shipped patch. No donor install/scripts executed.

## Ranked remaining excess

After the previous8port picks: actual component means26d5054f, original
seeds' actual lengths, unique full KV rows or sliding union2048+L-1,
1.3x bytes/4.8TB/s. R uses S19 CUPTI9c16c850 non-attention wall with the
measured component replaced. This is a **ranking scenario**, not measured
G0 or an attainable guarantee: fixed launches, softmax and norm still cost.

| Rank | Cell | Pick us | 1.3KV us | Excess us/layer |46S/16F excess us/load| Six-load R if removed |
|---|---|---:|---:|---:|---:|---:|
'''
for i,q in enumerate(sizing['rows'],1):s+=f"| {i} | {q['name']} | {q['time_us']:.3f} | {q['target_us']:.3f} | {q['excess_us']:.3f} | {q['weighted_us']:.3f} | {q['R_if_removed']:.6f} |\n"
s+='''
## Arms and measured verdicts

F consumes FP32 split planes directly, rounds their sum BF16, then existing
OP both-head/last-arriver (d1S/v1S) or Lee native math. J is direct-input Lee
fdec atd1S. O is direct-input OP atd16S/d32S/v16S. K applies Lee keys-on-M
attnd to full decode16/32. P moves V widening into producer; Q loads query/
own-KV early; E uses25% polynomial/75%MUFU exponent issue; H combines P/Q/E.
Each is a real timed kernel, whole context/window, no free-value oracle.
BF16/cache precision and external true-gate ownership unchanged.

Ratios below are **new/post-port pick**,95%paired-seed t7. Best tested arm
shown even when it loses; all33arm points, SEs, seeds and P/Q/E/H-vsF
contrasts are in `result.json`. Raw reconstructed rounds: `receipt_data.json`.

| Cell | Best arm | New/pick[95%CI] | A:A[95%CI] | Nomination |
|---|---|---:|---:|---|
'''
for cell in r['controls']:
 p=best[cell];s+=f"| {cell} | {p['arm']} | {f(p)} | {f(r['controls'][cell])} | {'P4' if p['nominated'] else 'retain pick'} |\n"
s+='''
V1F A:A .993350[.987504,.999196] has−0.665% directional drift; disclose
that uncertainty, no per-load veto. Every other control CI covers1. No v1F
winner or precise small-effect closure. TASK is the correctness gate.

**Two nominees**: v1S F .839438[.819888,.858987],2->1launch,
9.513->7.983us; d32S F .985855[.960184,1.011527],2->1launch,
14.635->14.426us. The latter's CI crosses1; P4 still gets its correct point
nomination and confirms held-out whole-layer G0. No whole-layer R measured.

Generic direct-input Lee loses heavily elsewhere: d16S2.805, v16S3.059,
v1F1.402, v16F1.379. The source performs scalar component-inner/plane-inner
loads (8 scalar float loads per8BF16values per plane, versus one old128bit
load), division/remapping and runtime plane loops. Repeated query bytes alone
at v16S are25.17MB versus4.19MB old BF16 queries and~8.4MB KV; d16S10planes
read10.49MB queries versus.52MB old. Extra bytes do not explain the entire
loss; per-phase issue/occupancy attribution was not measured. Some verify
prologues spill in native SASS. This **closes this generic scalar reader**,
not direct partial fusion; vectorized plane-outer loads remain unmeasured.
OP uses compile-time tensor loads and wins v1S; at d1S F still loses1.069,
so its remaining gather/last-arriver fill-drain is the largest open term.

Producer-V costs1.035–1.199 relative to F on tested cells. Q is .98645 at
v32F relative to F but still1.0663 versus S19; E is .9941 atv16S relative
to F but still3.0395 versus the pick. These do not nominate transplanted
original-port features. H loses1.169–1.250 vs F. Full K loses1.342 atd16F,
1.196 atd32F. All scope to these exact flags/reader/gather/split geometries.

## Exact P4 handoff

`P4_HANDOFF.json`: combined patch **9b043487**, source **fbaf2793**, on
P4 current S21 shipment **eacc087b**, expanded109c0135. PLAN's P4 row still
names the olderd5c6dd24; `base_contract.json` pins the actual current file
and manifest. S21 keeps C3 atd16F and Rw4 atv32F; neither changed handoff
cell has that comparator difference. No base candidate edited by this lane.

Independent v1S-only patch80e45c72/sourcee5931a6a and d32S-only patch
a59cb4b2/source13ea0f74 are alternatives, not stacked prepare wrappers.
Unselected experimental families are pruned: Lee128KB->35KB, OP44KB->19KB.
Selected OP kernel/helper ASTs identical; selected Lee has80registers,
no spills,7024SASS words identical except one equivalent shared-symbol
alignment immediate. Both expressions map to shared base2048, same one-CTA
occupancy; `selected_validation.json` keeps exact word/resource proof.
Final selected wrapper normal/forced:28cases eager+poison/repeated graphs
(56TASK each),6static pass. Adds setup to S21's prepare pool before warmup;
actual full prepare<=300s and held-out whole-layer G0 remain P4 confirmation.

First native1c5f639111fa4f80 is **unmeasured**: all cells reached, but its
13150-byte result exceeded11200. Repaired exact-GCD/LEB128+XZ transport
round-trips H200/local/adversarial integer streams with SHA256 and retains
every seed/arm/round; no tests or kernels changed. `codec_validation.json`.
Repaired sender948ed88bc3c00a01/payload59cfd432 is done; nothing queued.

Rebuild: public_ports build.py/build_lab.py, then here build.py/build_lab.py;
normal+forced screen.py via judge image; analyze.py receipt; prune.py;
build_candidate.py --base <S21/S22>; native compile selected, both
screen_candidate.py modes and check_selected.py. Preserve license notices.
'''
(H/'README.md').write_text(s)
# One row per cell's decisive tested recipe; full per-arm points stay linked.
p=R/'docs/FINDINGS.md';text=p.read_text();mark='### P5b post-port direct-input and consumer hybrids';end='## Projections and verify dense'
block=mark+'''

Post-port picks26d5054f,8freshseeds10739123+7919*i, all4752 cold integer
round times retained,2N cyclic/reverse rounds,780nativeTASK pass.95%t7,
no filtering. All A:A CIs cover1 except v1F .993350[.987504,.999196];
disclose drift, no precise marginal closure there. No whole-layer G0 claim.

| Recipe | New/pick [95% CI] | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
'''
for cell in r['controls']:
 q=best[cell];mech='Direct FP32 sum/BF16 input round inside OP both-head/last-arriver or Lee native prologue' if q['arm'] in ('F','O') else 'Lee K-major native FP8 QK/f16 PV on full decode' if q['arm']=='K' else 'Lee direct-input + early Q' if q['arm']=='Q' else 'Lee direct-input +25% polynomial exponents'
 leaves='P4 nominee, held-out whole-layer/prepare confirmation' if q['nominated'] else 'Retain pick; closes this exact reader/geometry, not the mechanism family'
 block+=f"| Post-port {cell} {q['arm']} | {f(q)} | {mech} | {leaves} | `cde88b9136de4093` |\n"
block+='''
Largest modeled remaining cells to1.3KV: d1S347.83us/load/R1.0121;
v1S432.79/1.0095; d16S506.92/1.0073; d1F198.00/1.0068;
v16S745.19/1.0064; v1F272.50/1.0059; v16F667.48/1.0057.
These combine post-port component times with S19 CUPTI9c16c850 non-attention
wall and actual old-seed KV draws; ranking counterfactual, not measured R.
Full d32F is last41.45us/load/R1.0005 despite180us kernel time.

Generic scalar QView loses2.805/3.059 at d16S/v16S: repeated FP32 queries
10.49/25.17MB versus oldBF16 .52/4.19MB, scalar per-component/runtime-plane
loops and some verify spills; exclusive issue/occupancy cost unmeasured.
Vectorized plane-outer reader remains open. Producer-V, earlyQ, poly25 and
combined lose against picks; some relative-F points improve but are not
measured original-port nominees. Full K-major loses1.342/1.196 d16F/d32F.

Combined exact licensed patch**9b043487**, source**fbaf2793**, onto current
P4S21**eacc087b**/expanded109c0135; v1S-only80e45c72, d32S-onlya59cb4b2.
Current P4 manifest/source newer than its PLAN row; C3d16F/Rw4v32F comparator
differences do not affect changed cells. Only selected families ship, same
OP kernel AST and LeeSASS except proved equivalent shared-base relocation,
Lee80regs/no spills. Final normal/forced28eager+graph cases each pass.
P4 confirms actual full prepare and held-out G0. First1c5f639111fa4f80
unmeasured due13150-byte result; exact timestamp codec repair retains all
samples/no gate changed. [Evidence](../problems/muse_attention_decode/dev/public_hybrid/README.md).

'''
if mark in text:
 a=text.index(mark);z=text.index(end,a);text=text[:a]+block+text[z:]
else:text=text.replace(end,block+end,1)
p.write_text(text)
