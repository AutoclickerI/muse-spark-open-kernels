# Generic memory compression (P10)

H2004233660b1bff4d18 landed: support1, create success, granted type1. Zero/pattern
bandwidth gains2.0265x/1.7008x prove activation. Random/raw-NVFP4/BF16/recoded
gains are0.9971/0.9991/1.0024/0.9977x. NT1 M1/M8 and QKV/O latency ratios are
1.0025/1.0009/0.9996/1.0024: no useful weight or kernel gain. No compression
drop-in was emitted or enabled. This closes the measured hardware placement
recipe for these banks; it does not rule out software coding.

Both incumbent MoE cases vary even in H200 A:A (M1:10/24; M8:16/24). W2 and
ordered outputs are exact across allocations, and QKV/O outputs are exact.
`h200_result.json` retains all raw failures and the complete receipt text.

Goal: determine whether driver-managed lossless memory compression helps the
overall decode-layer score. One MoE-hosted lab measures support, real data and
the actual incumbent kernels. The host prepare/state and scored function are
unchanged; no compressed-memory policy is enabled in best7_scale.

`cuda.bindings.driver` is installed in the judge image. The probe records the
device attribute, requested COMP_GENERIC, allocation-granularity status,
`cuMemCreate` status and actual `compressionType` returned from
`cuMemGetAllocationPropertiesFromHandle`. The allocation wrapper then reserves
VA, maps the handle, grants device access and wraps its pointer through the CUDA
array interface. The resulting torch tensor must alias the exact pointer.
Normal and compressed arms use the same VMM path with requested types0 and1;
every large allocation's granted property is checked separately.

The allocation owner has an explicit `close()`: retain it with the weight state
and close only after all GPU users have finished. There is no GC-time unmap or
timed allocation. BF16/FP8/FP32/uint8 tensor views preserve their exact bytes.
Copies use actual vector SM stores to the mapped allocation; they are verified
byte-for-byte before benchmarking.

Six256MiB read controls cover zeros, a repeating FP32-one word, seeded random
bytes, the host's raw NVFP4 codes plus scales, seeded BF16 N(0,0.02), and the
host's recoded NT1 bank. Raw sampling retains W13-code/W13-scale/W2-code/W2-scale
proportions16:2:8:1. The recoded control includes both W13 and W2 tiles/scales.
Three streaming configurations use512 threads with8/16 independent128-bit
loads per thread and grids1/4/8 times the SM count. The fastest **normal** reader
selects the reported pair; all three bandwidth ratios are printed. Volatile
loads consume every byte, including zero/pattern data. The same stream outputs
must match bit-for-bit.

Every graph replay follows an uncompromised256MiB flush outside the events.
Controls use12A:A and24A:B pairs per configuration. Kernel effects use24A:A and
36A:B AB/BA pairs across three identical seeded input/route banks. Logical read
GB/s includes the common graph envelope. The zero/pattern controls demonstrate
activation if they gain; a zero null is explicitly inconclusive.

Kernel cases are best7_scale's exact recoded NT1 at M1/8, plus its M1 QKV and O
cuBLAS choices. Real host experts are losslessly repacked by the frozen
best7_scale packer/reader; the recoded bytes never go to a raw-E4M3 reader. One
mapped bank holds every immutable expert tensor, including both global scales
and postnorm, while synchronization counters remain normal memory. QKV/O use
fixed-seed BF16 N(0,0.02) banks with their real shapes. Actual weights and inputs
remain identical across arms.

There is a necessary qualification to the requested final-output bit gate:
the incumbent uses unordered FP32 `atomicAdd(ss + off + idx, t2)` for its norm.
On the local sm120, M8 already differs in A:A; the first discovery screen found
16/24 A:A and19/36 A:B mismatches. That failed bit gate is preserved in
`incumbent_nondeterminism.json`, not relabelled as a pass. Incumbent timings
remain untouched and print raw bit-mismatch counts. A separate **untimed**
control exposes the same module's W2 outputs and applies an ordered norm/combine:
all valid W2 values and final ordered outputs must be bit-identical between
allocations over12 fresh replays. It must also agree with the incumbent within
0.1% relative L2. Dense outputs retain the strict final bit gate. This isolates
allocation correctness without claiming the atomic incumbent is deterministic.
No MoE case with a failed raw bit gate is advertised as bit-exact.

NVIDIA documents ILC as a bandwidth feature that preserves allocation footprint;
these are effective bandwidth ratios, not measured physical compression ratios.
Primary sources: [Hopper inline compression](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#inline-compression),
[allocation compression properties](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/structCUmemAllocationProp__v1.html).

Local sm120 evidence: support and granted type1; zero/pattern readers gain
about3.4x, while random/raw/recoded/BF16 controls are near parity. All six stream
outputs, all mapped bytes, dense outputs and ordered MoE controls are exact.
Both MoE W2 controls are exact; the ordered/native relative difference at M8 is
about1.8e-5. The five host tests and six static checks pass. Three native sm90a
transport kernels and two ordered-control specializations compile, with no
native spills. These are local findings, not H200 claims.

`sent.json` binds the sent source and baseline to the payload; `read_receipt.py <job>`
reads the landed result.

```bash
python problems/muse_bf16_proj/dev/memory_compression/build.py
python tools/h200_lab.py build prof/lab/generic_memory_compression.py --host muse_moe_w4a16 -o problems/muse_moe_w4a16/lab/lab_generic_memory_compression.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/memory_compression/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/memory_compression/audit.py
python tools/h200_lab.py send prof/lab/generic_memory_compression.py --host muse_moe_w4a16 --priority 0.5 --note "P10 generic compression support, six read controls and incumbent NT1/QKV/O effects"
```
