# S14 final prepare-tensor audit

Final **f1c77ce6**, exact base S13 final **958f1258**. Normal/forced local_eval pass 13 tests,
12 graph shapes and 24 static checks each. Recursive inventories in `stage3_screen/normal/` and
`stage3_screen/forced/` retain 78/118 tensors, including adapter objects and closure cells.

| Retained bank | Encoding / precision |
|---|---|
| coded_dense.* | S13 original lossless v3 BF16 codes, exponent-window FP32 metadata and original-bit escapes |
| s5_hh_base w13/w2, *_sf | Packed E2M1 codes in int32 words plus byte E4M3 scale codes, losslessly permuted for PERMW; 9 bytes per 16 expert weights |
| Native NT1 / adapter banks, global scales, norms | S13 formats; global expert scales FP32, clones preserve precision |
| nf_streams | Five stream/event handles; runtime scratch/output buffers overwritten on every replay |

No decoded BF16/FP16 expert prepare bank is retained. All weight/scale decoding stays in timed kernels.
Light 9..16-token experts execute measured N16G4 W13/W2 once on a third stream; NT1 handles <=8,
N32 handles larger experts, and both side streams join before finalization. No routes/contributions are
skipped. The measured bundle includes PERMW at both M128 and M256 and removes S13's original M128
module/bank. The audit checks every expert bank's packed dtype and nine-byte-per-16-weight storage.

Final bal2 dispatch is limited to pool/router T32, BK128/NS2/S6. Original coded T1 remains. Graph dispatch
checks observe this selection; all untouched shape controls are measured in the whole-layer A/B.
Source/patch hashes and exact final-vs-admitted-source equality are pinned in `final_admission.json`.

The rejected initial trial's inventories in `normal/` and `forced/` additionally cover direct BF16-coded
qkv3/O. Its banks were independently cloned and bit-exact decoded with integer origin/base checks.
Those banks are absent from the promoted final. Historical inventories remain evidence for the first
H200 send, not a description of final prepare state.
