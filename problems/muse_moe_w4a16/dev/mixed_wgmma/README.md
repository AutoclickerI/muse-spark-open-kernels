# Grouped NVFP4 RS WGMMA

Owner: the free lane. No H200 win is established yet. No commits or candidate sends by this lane.

`engine.py` uses TileLang 0.1.12 to put output channels on WGMMA M and routed tokens on N.
Two GPU routing kernels form a padded prefix and hot-first order; a persistent grid processes
channel tiles and splits hot experts along N. There is no inter-CTA spin barrier. Both GEMMs
convert E2M1 and raw E4M3 group16 scales inside their timed mainloops. FP32 per-expert global
scales are applied after FP32 accumulation. SwiGLU rounds to BF16 before W2; norm/reduction
keep FP32 arithmetic. Returned state contains only original or permuted packed weights/scales.

Two layouts are tested: ordinary nibbles with a PRMT lookup, and a bijective bit-plane permutation
matching RS fragment ownership. The latter reuses the incumbent V9 extraction table, then uses
packed BF16x2 multiplication. Its temporary 2^-8 scaling is undone in the FP32 GS epilogue.
This changes the new WGMMA producer, not V9's dequant/schedule (the MoE lane owns that work).

## Evidence

- Scalar arithmetic: 130,048 finite packed-byte/E4M3/half combinations, bit exact, including
  signed scales, zeros and subnormals. Plane arithmetic: 1,040,384 finite slot/code/scale cases,
  bit exact. `validate.py` reproduces both.
- Scalar producer: four configurations at M8/128/256; 12 eager +36 refreshed graph checks against
  independent FP32, relative L2 <0.001. Empty routing and one-expert routing pass. Eight generated
  W13/W2 kernels compile for sm_90a, with RS/SS WGMMA present in source.
- `cand_r3_mixed_wgmma.py` SHA256
  `4b3753823bd6176bda1abc2b373e87ebeb32418c7e941e323525674081338dcb`
  passes 13/13 official layer tests and five static checks, prepare108.2s on SM120. Forcing two
  plane producers through all four verify16/32 full/sliding cases passes24 fresh-input graphs
  (three refreshes per case/producer). Its incumbent is the MoE lane's raw-E4M3 V9. The local
  timing gate retains V9; no H200 ranking is inferred.
- The scalar lab passes all six M sizes locally, every FP32 gate, 9/9 proj-host tests and five
  static checks. Local prepare 238.2 s, including all lab experiments. The wrapper's optional
  state parameter was made explicit to satisfy the call-signature check; runtime math is identical.

## H200 jobs and protocol

`prof/lab/moe_mixed_wgmma.py`, SHA256
`9dca1a1e70e56716da341774ba6571ff1fae86dc9c5bd2dfb22080aa7c597090`,
was sent through `h200_lab.py` on proj. Judge job **296fdf45382244dc**, source key
`5d16a813364a8d20`. No receipt was polled or awaited after sending.

`prof/lab/moe_mixed_wgmma_planes.py`, SHA256
`224f4cdd6189de1d5db6e07e13bba6f125fac6292f0cbbfe3b3f90d02da67f6a`,
is queued on proj under source key`71c2f471b2e74efe`. It tests N16/N32 at core sizes and
N64 at prefill sizes, every required M. The final payload passes all six M locally,9/9 host
tests and five static checks; warm prepare31.7s. An earlier wider tile sweep exhausted the
prepare budget before16K, so it was replaced before sending. No correctness gate was relaxed.

The plane family independently passes12 eager +36 refreshed graph FP32 checks at M8/128/256,
plus six extreme-routing graph refreshes; all eight W13/W2 kernels compile for sm_90a.
Conversion and graph evidence are in `runs/screen/` and the local ignored logs beside this file.

`prof/lab/moe_mixed_wgmma_prefill.py`, SHA256
`042c8ef6d094439c2e3466b583bcff946143fc324e83906baf5d8ca7d9cbefd3`, source key
`9176ba0035ba0564`, compares N64 with Marlin and the prefill lane's raw17, masked Humming and
dg_w8 at512/753/3012/16384, on identical fresh inputs. `build_prefill_lab.py` extracts those
producers from the source named/hashed in the payload, with gather-only adapters for normalized x.
Local plane/raw17/masked arithmetic passes all12 three-input FP32 gates;9 host tests and6 static
checks pass. DG's Hopper-only Python launch ABI is forced locally with an explicit zero-output
stand-in, never a timing contender or arithmetic claim; the real DG path runs only on H200.
Warm local prepare10.6s; H200 compile time and arithmetic remain unverified.

The lab uses exact fixture0 IDs exported by 26d7e05 for M128 (209 pairs/18 experts/hot55)
and M256 (414/21/119). Its fresh activations and weights come from the complete public decode
generator at seed5353; the original captured job's seed/weights are not claimed. Larger M512,
753,3012,16384 use that public router with q_len1, not proj's iid router. Three fresh datasets,
six alternating cold graph pairs, 128MiB L2 writes; FP32 gate L2<=1%, row<=5%, finite outputs.
V9 and Marlin controls run in the same lab; only Marlin is used beyond V9's fused routing
capacity. Raw GPU microseconds include graph overhead. Phase graphs are diagnostic and do not sum.

Build payloads with `build_lab.py` or `build_lab.py --planes`; screen/send via `tools/h200_lab.py`.
`build_candidate.py` creates the standalone layer adapter, with shape-only choices admitted by
three FP32 checks and cold paired ratio <0.98. The owner screens/sends candidates and commits.
`read_receipt.py <job>` renders local H200 receipts without querying the API.
The `*.py.inc` files are assembly fragments; undefined-name checks run on every standalone
generated candidate/payload, where the imported/copied names are defined.

## Serving producer

If the new producer wins, it replaces the expert GEMMs inside `muse_spark_moe._grouped_call`.
The serving hook receives already normalized x: use gather only. Export per-pair W2 outputs
before local weighted reduction and retain the existing EP collectives and output-norm rounding.
Serving gate/up GS can be independent, whereas the board's W13 GS is shared: expand the FP32
epilogue GS vector by output row when porting, never fold it into E4M3 or packed codes.
Serving wiring, TP agreement and H200 end-to-end performance remain unverified.
