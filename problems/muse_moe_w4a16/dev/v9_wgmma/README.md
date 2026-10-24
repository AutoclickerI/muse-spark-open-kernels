# V9 raw64 WGMMA decode variant

`raw64_dropin.py` specializes the prefill lane's raw17 register-A WGMMA
pipeline for decode-shaped expert counts.  It keeps NVFP4 E2M1 codes and E4M3
block scales packed through `prepare()`, repacking only their layouts.  The
timed CUDA path converts both representations in registers and fuses W13
SwiGLU before the W2 projection.

The specialization changes the token tile from `m64n128k16` to `m64n64k16`.
The route gather and combine use the same 64-token tile width, so an expert's
usual 12–20 routed rows share one weight tile while hot experts split into
additional 64-row tiles.  The CUDA extension retains the 128-token raw17
control for comparison and exposes both kernels from one packed-bank module.

Local verification:

- `runs/screen/moe_decode_adapters.json`: 156 fresh graph gates; raw64 whole
  output L2 is about 0.0048 and every max token-row L2 is below 0.0056.
- `runs/screen/raw64_dropin_static.json`: five challenge tests pass, undefined
  names and call signatures pass, and the raw17/raw64 CUDA source compiles for
  SM90a.
- SM90a ptxas reports zero spills and zero stack bytes for raw64's four
  MODE/STAGES instantiations; native `HGMMA.64x64x16` disassembly is saved in
  [`raw64_sm90a.sass`](raw64_sm90a.sass).

The H200 comparison is the lab submitted from
`prof/lab/moe_decode_adapters.py`, with raw64 added beside NT1/NT2/raw17,
masked Humming and DG.  It uses exact R0 plus the same 12 public seeds and
independent FP32 gates.
