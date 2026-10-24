# Projection handoff: CUTLASS H200 lab pending

Solo/GPU0/no commits. Never poll judge. Check once per turn; end if only receipt remains.
Do not resend frozen sources. Lab host's scored path remains proj f3786341611f4f1f.

|Order|Source|SHA256|Status|
|---|---|---|---|
|1|`prof/lab/verify_proj_cutlass.py`|`e43ed9b225d98cf1feaf1b720f295ca21a69bf74a389cc560ed4ffc588246bab`|sent priority0; detached log `runs/detached/b81c9564402fa62c.log`|
|—|`lab/lab_verify_proj_cutlass.py`|`b81c9564402fa62c4e4a87cca541c853506aa6884c8cd4b41b61f583b7fb34ab`|exact screened host9/9+30/30 graph shapes;3 static checks pass|

CUTLASS lab: M128/256 qkv2304x8192, pool/router4608x8192 FP32, unpool8192x4096,
o8192x1024. Eight native tiles×2 dtypes;144 recipes/key, standard Hopper cooperative
TMA/WGMMA and CUTLASS Stream-K scheduler. Explicit splitK2/4/8/16, cluster multicast,
swapped operands, raster/grid/reduction choices. Counter reset occurs in each timed call.
Three fresh inputs (one-column probe), independent FP32+incumbent1% gates, cold1GiB
float32 L2 flush, CUDA graphs, swapped order,20-pair finalists.225s lab deadline;
actual cold H200 compile/prepare total must be verified in LAB_END.

Local **sm120**:9/9 tests+30/30 fresh graph shapes. Every source compiles SM90a.
`dev/cutlass/validate.py` forces full Hopper module/ctypes creation,1152 recipes and
both dtypes across all8 shapes, including capture/finalists.48 fresh/guard integration
checks pass; AttributeError/runtime/accuracy/budget/unknown-tile failures retain incumbent.
Portable math verifies glue; no Hopper timing is inferred. Logs in `runs/screen/verify_proj/`.

Drop-in `dev/cutlass/projection_cutlass.py`; API/merge instructions in `dev/cutlass/README.md`.
`vgc_bind(incumbent)` preserves `project(x,w,name,state,out_dtype)->(partials,splits)`.
`vgc_prepare(state,incumbent,deadline,recipes)` retimes H200-positive recipes only.
Default empty before receipt; incumbent holds elsewhere and on any optional exception.
Use `dev/cutlass/read_receipt.py <receipt> --recipes <file>` once the receipt lands.
It checks the frozen LAB_BEGIN payload SHA, reports controls and emits positive finalists.
Do not hand a mocked winner to decode or serving. Retime actual winners in full layer.

Landed: library596b2a5d436e4fee, payload4afe9807…; M128 qkv21.104->20.224,
unpool24.000->22.960us, others parity. Exact7-layout old block SHA
97c874743a23ab6eef0beee32cbd153e436b193e811eaaaaab5dea78dac8450a,
`dev/verify/projection_dropin.py`; full-layer sm12013/13+12/12,4.877s prepare.
Custom TMA v2 job92e7802783a54212 loses every key1.21–1.95x; PDL chain loses too.
Do not resend it. V1 job6b72087f64584258 is invalid due live-barrier reinit.

Dense cand_v11_gemm still unmeasured/genuinely queued (d318167e is tma_cluster).
MoE tma_grouped is withdrawn/parked after H200 losses; exact loaded-weight Humming/V11
agreement fails, so their synthetic gains are not portable. #45015 is serving-lane-owned.
