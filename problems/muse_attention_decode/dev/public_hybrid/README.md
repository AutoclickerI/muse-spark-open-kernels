# Post-port attention hybrids

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
| 1 | d1S | 7.704 | 0.142 | 7.562 | 347.829 | 1.012087 |
| 2 | v1S | 9.551 | 0.142 | 9.409 | 432.792 | 1.009513 |
| 3 | d16S | 13.292 | 2.272 | 11.020 | 506.924 | 1.007342 |
| 4 | d1F | 17.922 | 5.547 | 12.375 | 197.996 | 1.006755 |
| 5 | v16S | 18.480 | 2.280 | 16.200 | 745.192 | 1.006424 |
| 6 | v1F | 22.578 | 5.547 | 17.031 | 272.500 | 1.005916 |
| 7 | v16F | 130.416 | 88.698 | 41.717 | 667.476 | 1.005741 |
| 8 | d32S | 14.756 | 4.544 | 10.212 | 469.760 | 1.005268 |
| 9 | v32F | 220.095 | 177.430 | 42.665 | 682.634 | 1.004413 |
| 10 | v32S | 18.950 | 4.559 | 14.391 | 661.969 | 1.004278 |
| 11 | d16F | 97.159 | 88.690 | 8.469 | 135.496 | 1.001926 |
| 12 | d32F | 180.006 | 177.415 | 2.591 | 41.451 | 1.000457 |

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
| 1x1s | F | 1.068873[1.056879,1.080867] | 1.004999[0.991366,1.018632] | retain pick |
| 1x1f | F | 1.292213[1.273897,1.310530] | 0.995619[0.986281,1.004958] | retain pick |
| 16x1s | O | 1.294585[1.276435,1.312734] | 0.996833[0.979222,1.014444] | retain pick |
| 16x1f | K | 1.341944[1.336264,1.347624] | 0.999752[0.996881,1.002623] | retain pick |
| 32x1s | F | 0.985855[0.960184,1.011527] | 0.994836[0.981340,1.008332] | P4 |
| 32x1f | F | 1.130155[1.125569,1.134742] | 1.002089[0.997109,1.007068] | retain pick |
| 1x8s | F | 0.839438[0.819888,0.858987] | 0.992761[0.959641,1.025882] | P4 |
| 1x8f | F | 1.402405[1.388335,1.416474] | 0.993350[0.987504,0.999196] | retain pick |
| 16x8s | O | 1.024066[1.016012,1.032119] | 0.999947[0.991950,1.007945] | retain pick |
| 16x8f | F | 1.379079[1.374989,1.383169] | 0.998869[0.997071,1.000667] | retain pick |
| 32x8s | E | 1.126020[1.108130,1.143910] | 0.991769[0.977067,1.006471] | retain pick |
| 32x8f | Q | 1.066325[1.060408,1.072243] | 1.000373[0.998426,1.002320] | retain pick |

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
