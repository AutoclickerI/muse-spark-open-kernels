# P5 sliding and decode-1 handoff

P4 receives **every measured per-shape point estimate below 1**. A:A,
across-seed SE and seed ranges describe uncertainty; they do not reject a
shape. P4 composes the pieces and promotes only when the held-out lower
confidence bound on per-seed six-load `R=exp(mean(log(T_old/T_new)))` exceeds
1, with `T=46*t_sliding+16*t_full`. Correctness is TASK.md's checker only.
Missing replay correctness and an established failed experimental control
remain unmeasured. Stage-only and older-base results retain that scope.

[point_handoff.json](point_handoff.json) is the complete P4 inbox: 42
point-winning arm cells, including duplicate/control cells, plus the E4/OLD4
incremental nomination. Every entry has ratio, SE, seed count, exact receipt
source and its SHA; each run contains its actual seed IDs. Recipes in the
round2/repeat archives include `[family, QT, splits]`. Exact historical global
and history shaders live in their immutable receipt sources.
[handoff.json](handoff.json) retains rebuildable integration selections.
[P4_decode_points.json](P4_decode_points.json) and its exact tracked patch
deliver screened decode16 D8 / decode32 D4 candidatec2000b3c without an
ignored-output dependency; apply onto exact S14, then P4 composes onto current.

Rebuild all estimates directly from receipts:

```sh
python3 problems/muse_attention_decode/dev/sliding_round2/rejudge.py
python3 problems/muse_attention_decode/dev/sliding_round2/report.py runs/muse_attention_decode/20261026T023034Z_check_58f0b30ba8b8472f.json
```

Ratios use the paired seed times retained in the archives (1ns quantization
where applicable). Consequently their last few digits can differ from the
unquantized `PAIR` log summaries. No local timing is native ranking evidence.

## Receipt verdicts under the current gate

| Receipt | Current disposition |
|---|---|
| `ae6cd27877cc4ee2` | Unmeasured whole-layer replay: no TASK gates. No nomination. Repair below. |
| `b05713ecf6574ad9` | Nominate v1 C4/C8/C16 and v16 C4/C8 as stage-only evidence; no v32 point winner. Selected native whole-layer follow-up is complete below. |
| `402819c28279466d` | Reopen compact history: E4/OLD4 0.997414, SE0.001244, n16. Also hand all OLD4/E4/E2/L4 point winners vs A to P4. The old A:A/3SE rejection is withdrawn. |
| `6c26b14fdd28458b` | v1 remains unmeasured: identical C16 DOUBLE/COOP 1.03901, SE0.00300, failed the experiment's control. v16 QT1 cooperative recipes all lose. |
| `6a0b3f0987e94ba5` | Nominate RESET/COOP/HALF/DOUBLE at v1; C16 pooled0.952913, SE0.005071, n35. Duplicate control agrees; full TASK/cache gates pass. |
| `27a914f72bfb47c8` | Reopen d16 D8 0.995283, SE0.007209; d32 D4 0.989822, SE0.007613, and L4 0.993761, SE0.007368; n16. Hand all seven v1 point-winning arms to P4 too, including E8 and D8 previously excluded by 3SE. |
| `857f78fa889f4646` | No resident-history point winner at either verify load. Close only the tested schedules. |
| `bb7239afc1b3487a` | All replay TASK/cache gates pass. Hand all14 point-winning arm cells to P4 with S11 scope, including old controls. Full-decode1 global recipes still lose. |
| `58f0b30ba8b8472f` | Complete: n35, all seven arms/shape pass TASK/cache gates. Nominate d16 GLOBAL0.981291/ALIAS0.978600/LAST0.978764/PP0.992226 and d32 LAST0.999170. No d1 point winner in this cohort. |

These are point nominations, not automatic native confirmations or overall
score promotions. Historical receipts that passed the former 20% TASK margin
also pass TASK; their immutable sources/results are retained. New lab templates
use exact public TASK tolerances and keep output/route drift as diagnostics.
The reporter no longer applies A:A or per-load 3SE acceptance cuts.

## Integration artifacts and screens

Exact S14 base: `f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.
`runtime.py` is the full original matrix. `kernels.py` generates last-CTA,
early-history and cooperative DSM; exact old DSM is retained separately.
`layer_patch.py` ships only selected CUDA families and emits an exact-S14
candidate; P4 ports the selected changes onto its current composition.

```sh
python3 problems/muse_attention_decode/dev/sliding_round2/layer_patch.py --selection '{"16,1":["dsm",1,8],"32,1":["dsm",1,4]}' --out problems/muse_attention_decode/dev/sliding_round2/out/s14_p5_decode_points.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/screen_layer.py --candidate problems/muse_attention_decode/dev/sliding_round2/out/s14_p5_decode_points.py --out problems/muse_attention_decode/dev/sliding_round2/out/decode_point_normal
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/screen_layer.py --candidate problems/muse_attention_decode/dev/sliding_round2/out/s14_p5_decode_points.py --out problems/muse_attention_decode/dev/sliding_round2/out/decode_point_forced --forced
```

Decode candidate SHA `c2000b3c4d64c292ccac0a61295df60d8478b232ed7e9fdf0de118806fa924f4`.
Normal and forced screens each pass13 TASK tests,12 graph shapes and24
static checks; exact hashes/evidence are in `point_admission.json`.
The existing verify1 DSM QT4/C16 candidate is `403bb778`, rebuilt with
`--selection '{"1,8":["dsm",4,16]}'`; its normal/forced screens each passed
13 TASK tests, 12 graph shapes and 24 static checks. On sm120, DSM uses an
explicit genuine C1 math stand-in. Native cluster geometry is measured in
the H200 receipts, not inferred locally.

`build_packet.py` builds the original matrix and shared public checker;
`followups.py` rebuilds that header from tracked inputs before rendering
selected/history/DSM/global packets. The builder supplies h200_lab's current
`print_line` wrapper field; the rebuilt exact-TASK global payload compiles
(SHA48ef4de6) and is unsent. Historical sent payloadb70ff21c stays immutable.
`check_variants.py`, `sass_check.py`, `screen.py`, `check_lab.py` and
`screen_layer.py` retain local admission tools. Generated artifacts go in
ignored `out/`. Frozen packet hashes/screens are in `admission.json` and
`response_admission.json`; they describe historical packets, not new sends.

## Latest collection and next work

Global confirmation wrapper `b899a8fb`, sender `1249027`, completed as
`58f0b30ba8b8472f` on attention check hosting. Full result:
[global_confirmation.json](../cluster_sliding/global_confirmation.json).
The old follow-up is collected; no P5 receipt remains pending.

The shared-`full[slot]` review item is audited against exact S15 `60254883` in
[operand_ready/README.md](../operand_ready/README.md). No active sliding
path contains that ring. Verify32 sliding is FA3, not VQ8; the confirmed full
VQ8 K/V split belongs to P5b. No inactive barrier arm or payload is built.
P4 owns the next composed six-load confirmation. Remote runs, if needed, go
only through `h200_lab.py`/`okc_run.py`; decode labs are checks, with packed
arms/A:A controls on an available host. Prefill hosting remains excluded
until its lab fix lands.
