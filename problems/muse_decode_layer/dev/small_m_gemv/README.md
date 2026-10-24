# M1 packed expert GEMV

Owner: prefill lane, decode-layer M1–32. This experiment targets decode1; its
H200 result closes this design: best57.07us versus NT1-raw44.17us. The baseline is the raw-scale NT1 bank from
`cand_r3_best6_small_nt2.py`. No candidate or serving dispatch uses this streamer.

The corrected H200 scale-contract lab `b741f04aaf53479d` passes every stage and
reports M1 NT1 47.44 us versus `newNT1` 48.00 us. That `newNT1` calls
`v9_small_nt1_kernel_fn` through the small NT4 module: it is the v9m1 body.
The next lab names it explicitly. NT4 loses at M16/M32 (149.18/141.39 us versus
NT1 93.68/110.75), so there is no NT4 promotion.

`gemv.cu` reads the existing canonical 36 KiB tiles directly. E2M1 bit-plane
rotation and stored E4M3 conversion happen in registers; dequantized weights
are never retained. Every touched M1 route receives one W13 and one W2 bank
traversal. W13's eight gate/up channel pairs cover all 4096 K values, with a
complete CTA tile issued before consumption, and write BF16 SwiGLU activations.
W2 partitions each 16-output tile over K1/2/4/8. Per-tile completion counters
select the reducer; per-expert and per-token counters select norm and weighted
sum finishers inside the same kernel. No CTA spins or grid barrier is used.

The fixed physical grids cover all 132 H200 SMs. Each CTA derives a compact
local-slot list from the current IDs and strides through its tasks; grid zero
launches one CTA per potential task as a control. W2 uses twice W13's grid.
The PDL variant reads W2 into registers before its dependency wait, following
[NVIDIA's PDL contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
The ordinary and PDL variants are compared; execution overlap is not assumed.

`prof/lab/build_moe_m1_stream.py` embeds the candidate's current raw-bank
implementation and this source, and writes `dropin.py` plus the self-contained
prefill-host payload `lab_moe_m1_stream.py`. The lab needs the host's loaded
router, bias, pooler and norms. It measures NT1, explicit v9m1, and fourteen
streaming recipes on twelve refreshed real-router input seeds, with independent
FP32 gates, rotated/reversed graph order and a 128 MiB cold-cache flush.
It reports paired medians per input seed and fixed-recipe means last. These
are twelve route draws of the host's weights, not twelve weight-bank seeds.

Local admission: 192 inverse bank tile checks, 160 forced Hopper ABI/graph
calls, and 720 native graph checks across five grids, four K splits, both
launch modes, six routing patterns and three refreshed activations pass.
Worst FP32 output L2 is 0.244%; negative, zero and subnormal stored scales
are included. The local real-router entry completes all twelve draws. Local
timings do not beat NT1, so no latency win is established. Logs and JSON are
under `runs/screen/moe_m1_*`.

The native launcher uses device pointers and the current CUDA stream through
pybind11, with no implicit PyTorch C++ headers. This removes the local compilation
bottleneck before a shared lab; the final local admission including the cold
build took 31.95 seconds versus 125.11 seconds with the tensor-based launcher.
The CUDA computation is unchanged by that host ABI replacement.

Rebuild and local admission:

```bash
python prof/lab/build_moe_m1_stream.py
CUDA_VISIBLE_DEVICES=1 tools/img python prof/lab/check_moe_m1_stream.py
```

For integration after an H200 win, inline `dropin.py`, compile
`m1_stream_module()` in prepare, and store it as `bank['stream_mod']` alongside
the existing `w13p/w2p/gs13/gs2/pnw` tensors. `m1_stream_kernel` supports M1
only; the winning split, grid and PDL setting must come from the paired receipt.

H200 lab accepted: **963515d3f0c04d91**, send key **7c830c8447f4c705**.
The immediate detached-log read showed no rejection. Every recipe loses (best P2g26457.07us versus raw NT1 44.17us);
[admission.json](admission.json) pins the source hashes and local checks.
