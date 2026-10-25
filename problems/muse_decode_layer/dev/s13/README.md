# P4: S13 final integration

Frozen: `problems/muse_decode_layer/cand_r3_best7_s13_final.py` **958f1258** on exact S12 final
`cand_r3_best7_s12_final.py` **240248b2**. Retain direct-partials attention at decode1 and PERMW at M256;
drop cpall coding and PERMW at M128. Original S12 v3 coding and F4 stay. No scored send from this lane;
the orchestrator owns the 00:00 UTC scored handoff and should now point it at S13 final.

`final_patch.py` is the aggregate handoff onto exact 240248b2; it asserts the output SHA.
`build.py --without cpall --out problems/muse_decode_layer/cand_r3_best7_s13_final.py
--manifest problems/muse_decode_layer/dev/s13/final_manifest.json` reproduces it (one command).
The builder pins every patch/dependency. Its default all-three intermediate is **4bb0fdd5**;
`patch.py` reconstructs that experiment, not the frozen handoff. Each experimental piece remains removable.

## Final admission and selection

`final_admission.json` / `final_manifest.json` pin the exact final file. Normal and forced each pass
**13 tests, 12 fresh-input graph shapes, 25 static checks**, including separate original M128 and PERMW256
Hopper CUDA. Retained tensor inventories: `m256_nocp_screen/{normal,forced}/precision_inventory.json`,
85 / 125 tensors. [Precision audit](PRECISION.md): original BF16/FP8 boundaries, lossless code/byte-scale
banks only, independent bit-exact BF16 decoding. The final patch reproduces the admitted file byte-for-byte.

The complete final wrapper **0c627bd2** passes local 12-shape/16-seed lab, all output gates, 13 tests,
25 static checks; decode1 has **32 rotating cold rounds**, other loads 8. H200 **cb38a0503a434573**:
official PASS, host **1.71859848x**, all 12 shapes and all seeds complete, no diagnostic failures.
Reverted/final ratios (>1 favors final):

| Selection | Paired ratio | A:A bound / decision |
|---|---:|---|
| Attention off, decode1 | 1.037688 | max absolute A:A .021515; wins |
| PERMW256 off, verify32, final run | 1.000923 | inside A:A .002260; neutral in this run |
| PERMW256 off, three-run geometric pool | **1.003756** | worst absolute A:A .003304; wins |
| PERMW256 off, three-run median | **1.004031** | also beyond worst A:A |
| S12 coding off vs S12, decode1 | 1.017397 | S12 A:A .010433; wins |
| S12 F4 off vs S12, verify16 | 1.005672 | S12 A:A .002263; wins |
| S12 F4 off vs S12, verify32 | 1.004903 | S12 A:A .002393; wins |

Every unchanged final load lies within its A:A envelope. S12's unchanged controls also pass.
The final single-run gate remains false for neutral PERMW; it is not overwritten. The retained PERMW
piece passes the **pooled** gate in `pooled_permw_report.json`. All three verify32 comparisons are valid:
the earlier split-QKV bug affects only T<=32, not T256. We independently verified identical host and
reverted NF_CUDA, NF_CPP, M256 launcher, and canonical packed-bank expressions across all three receipts.
All three M256 ratios are positive; geometric pool 1.003756 and median 1.004031 both clear the largest
observed A:A radius .003304 with margin above printed precision. This uses the same envelope criterion,
not a relaxed single-run check. `cb38a050_report.json` preserves the individual result and S12 subpairs.

## H200 jobs and rejected pieces

| Job | Host / score | Whole-layer result |
|---|---|---|
| **3dbcaccd1bb042a4** | 62eca97e / **1.74325594x** | PERMW off 1.001426 / 1.006322 at verify16/32: M128 neutral, M256 wins. Decode1 comparisons confounded by the inherited choice bug below. No freeze. |
| **4de869d41148400e** | 4bb0fdd5 / **1.72362918x** | Corrected choices. Attention off 1.031672; cpall off 1.010536 raw, **0.996850 A:A-normalized**: reject cpall. PERMW256 off 1.004031 wins. All unchanged controls pass. |
| **cb38a0503a434573** | 958f1258 / **1.71859848x** | Final conditional composition; selection and S12 recheck above. 16 seeds, decode1 32 rounds. |

Raw host scores do not rank these sources; paired whole-layer evidence selects the final.
Receipts/sent sources are in `runs/muse_decode_layer/`; profiles are beside the labs. Reports and archived
manifests are in this directory and `trials/{all3_62eca97e,m256_cpall_4bb0fdd5}/`.

Cpall's standalone c446fdab win did not survive the corrected layer comparison. The native body is kept
in `cpall_frozen.py` as experiment provenance, excluded from the final. PERMW at M128 was neutral, so M128
uses its exact original F4 CUDA/module and original bank; M256 gets the wide/rotated bank and module.

## Harness correction and request envelope

The inherited S12/S13 reverted namespaces reseeded only `_seed_v9_gemv_choices`, losing the QKV3 entry
that real `_gate_split_prepare` adds afterward. That made unchanged gate-split rows slower and inflated
decode1 reversion gains. Fixed in both S12/S13 harnesses: assert identical balanced/split-QKV settings,
then independently copy the complete prepared choice dictionary. Corrected controls pass. No gate weakened.
The added S12-nocode/noF4/S11 arms and separate S12 A:A graph directly revalidate the frozen base.

Each arm has its own namespace and native module caches; representation changes get a bank rebuilt from
identical original weights. Unchanged small-M bank/counters stay shared. Full arm SHA reconstruction is
asserted before capture. LZMA bundles and AST-identical outer comment removal keep wrappers under the
server's **1,000,000-byte** limit. The first 1,001,892-byte wrapper was rejected before any job/quota use;
the final wrapper is 919,421 bytes, with its exact size/hash recorded in its manifest/admission.

`../s12/compile_cache.py` reuses only successful exact CUDA/flag units from S12's uncached normal admission
in the same judge image; changed sources compile with the original checker at concurrency 2. Runtime,
forced dispatch, Python static checks and graph screens always rerun. This replaces duplicate compiles
that exhausted host memory. No changed source is skipped.

## Component provenance and next work

- Attention: `patch_partial_port.py` / `partial_port.py`, P5 c114c35e: 0.97091 new/old at decode1;
  exact sincos, all keys, original BF16 boundary. Only its base assertion was adapted from S11 to S12.
- PERMW: frozen `permw_frozen.py` + `wide_frozen.py`, P3b b88d04fe: 0.9862 / 0.9787 at M128/256.
  The layer evidence selects only M256. `patch_permw.py` adds exact original M128 CUDA and a second
  lossless packed bank; both modules prepare before capture.
- S12 coding/F4/wide selection: `../s12/README.md`; corrected coding/F4 proof is in cb38a050.
- Attention-only **45586939** passes normal/forced 13/12/24 as a contingency; it was never sent.

Next: orchestrator moves the scored handoff to S13 final **958f1258**. For the next integration, rebase
and pair the dense lane's new direct-qkv3 patch **2a4b14c9** (currently asserts a70dd30c); optional O patch
**f703e6d5** needs its own whole-layer proof. Neither is part of S13 final.
