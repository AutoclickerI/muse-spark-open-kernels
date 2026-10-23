# Ready: DFlash serving compute diagnostic (G1/G2)

Solo, GPU0 only. The orchestrator sends and commits; this thread did neither.
No judge process was launched here. Existing sends/queue entries belong to the orchestrator.

## Send order: 1

**`problems/muse_bf16_proj/cand_dflash_diag.py`**

Frozen SHA256: **`9b4fae2fc29251d8279112f2cd1ff65899b2a249c7618af3fdd97132aa28715a`**

Scored prefix/dispatch: byte-identical `cand_fused_select.py`, receipt **f3786341611f4f1f**,
SHA256 `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9`.
Its original prepare runs first; the diagnostic returns exactly its state to the scored kernel.
The new file is self-contained, with no imports of our repository modules.

| Local admission (**sm_120**, GPU0) | Result |
|---|---|
| `tools/local_eval.py` | **9/9 tests, 30/30 fresh-input graph shapes**, prepare **45.854s** |
| DFlash diagnostic | **32 dense +16 glue rows**, each20 cold-L2 order-swapped pairs; no `DFB_REJECT`/`DFB_SKIP` |
| Serving unit tests | **7/7 tests, 464 exact graph comparisons, 464 stock-repeat checks**, observed stock spread0 |
| Hopper glue forced with portable stand-in | **14** skinny/TN/NN/stock-NN ABI cases pass |
| Hopper compile | **16** SM90 pointwise specializations +raw SM90a dense utility build |
| Failure/budget isolation | Injected TypeError and wrong result rejected/logged; expired24 groups skip; original scored state returned; zero active tuning streams |
| Receipt tail | All50 compact `DFB` lines occupy **5,619 characters**, printed last after compilation/trial rejection output |

Evidence: `runs/screen/serving_dflash/{tests.json,sm90_build.log,diagnostic_guards.log,
muse_bf16_proj.cand_dflash_diag.json,diagnostic_screen.log}`.
The screen JSON records the exact frozen candidate SHA. Local timings are **not H200 claims**.

The diagnostic times all local TP8 dense roles at M8/32/128/256 (head P7/28/112/224):
conv coefficients, gate/up, down, already split context FC, QKV, o, existing NN context-KV,
and draft head. The strided selector-hidden projection stays stock bmm/count0.
Each dense trial must be exact to stock on three probes. Coarse trials feed a20-pair final
comparison; failed variants are dropped/logged. The overall diagnostic deadline is225s from
prepare entry, leaving15s below the240s task limit; incomplete comparisons are skipped.
The final48 stock/new comparisons check exactness on every fresh pair. Numerical failure,
layout/launch error or budget exhaustion never raises into the scored projection run.

## Serving handoff

Only new SGLang source: `python/sglang/srt/layers/muse_dflash.py`.
SHA256 **`ecd7348666f1bf34d2ec0bd73084d7fa32b2d8ad812c331478fbf448f3bac65a`**.
It uses existing `muse_proj.py`, embedded utility SHA
`734d241ab5dc26e4e758bc01d39a73337ddfcec6672b6b3169c6b276eeb99223` in the diagnostic.
Use `DFlashLinear` after released weights load, before capture; preserve parameter registration
and collectives. Use `stock_nn=True` for the existing `[8192,1280]` context-KV layout.
`DFlashConv2` binds the original coefficient weight/base, preserving Inductor's FP32 first-product
FMA and BF16 output. Selector helper accepts already projected rank256 hidden; it does not
replace strided hidden bmm or the walk. `dflash_unary` preserves in-place FP32 semantics.
No source owned by the serving lane was edited. It wires only measured winning choices.

[Full inventory/model](../../../docs/reference/muse_dflash.md) and
[serving levers](../../../docs/reference/serving_levers.md#per-step-breakdown-and-unowned-work).
Three removed unary launches model **0.006/0.006/0.006/0.006ms per step** at C1/4/16/32.
Other dense/layout/conv/selector savings get zero H200 credit until this receipt. Cold synthetic
compute measurements do not price TP8 collectives/overlap, full-model agreement, KV capacity or TTFT.

Reproduce:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/test_muse_dflash.py --json runs/screen/serving_dflash/tests.json
CUDA_VISIBLE_DEVICES=0 tools/img env PYTHONPATH=$PWD/serving/sglang/python python serving/tests/compile_muse_dflash.py
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_bf16_proj/cand_dflash_diag.py --json runs/screen/serving_dflash/muse_bf16_proj.cand_dflash_diag.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_dflash_diag.py
python serving/tests/dflash_inventory.py
```

Builder `dev/build_dflash_diag.py` refuses to overwrite the frozen candidate. Build future
revisions under a new name after changing its output; never mutate a queued source.
The pinned stock `_grouped_conv`/`_score_edges` bodies come from a55ab7d5.

## Receipts and earlier handoffs

**`3afea80c527c4969` / cand_portfolio_all:** receipt read. `passed=false`, no speedup or
completed shape timings; prepare failed at router M128 with
`AttributeError: '_AllModule' object has no attribute 'old_hopper'` in a coarse graph trial.
The log contains router M1/8/16/32 trial choices only; these are raw tuning events, not judge
shape medians. The orchestrator owns that source and its correction; do not resend the failed hash.
Its failure does not affect this diagnostic, whose scored prefix is the proven fused-select source.

Standalone best evidence remains **1.181998x**, f3786341611f4f1f; the30-shape receipt table
is in FINDINGS. Decode `cand_r3_proj.py` (SHA235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e)
was accepted/committed24d577c and passed13/13 tests +12/12 graphs locally; that lane owns its
subsequent queue/merge. Serving `muse_proj.py` was accepted/committed60e7333; serving owns wiring.
For inherited candidate state use `runs/waiting`, `runs/pending`, `runs/sent` and receipts,
rather than the superseded per-PID snapshot formerly kept here. No cancellation was requested.

Next: read all final `DFB` rows from the H200 receipt, reject numerically unsafe paths,
choose by measured per-rung effect, and hand the winning choices to the serving lane.
H200 execution/cold compile time, full released-model agreement, TP8 serving speed/TTFT and
post-capture KV capacity remain unverified.
