# Next: read the GEMV/glue H200 lab; do not resend

Solo/GPU0/no commits. Jihoon sends decode candidates. Never poll judge; one receipt check
per turn, then build or end. The scored proj host remains f3786341611f4f1f.

|Order|Source|SHA256|Status|
|---|---|---|---|
|1|`prof/lab/verify_projection_glue.py`|`34143ac6134efe1f0d0c98ae95e0d3d25425fa536339b734c80f5c190a88e0ac`|sent priority0; local request468733, judge job unassigned at local check|
|—|`lab/lab_verify_projection_glue.py`|`4b4182cb1d0ff580916f7ed6fbe3a9b236264917bea014fa7a09f111168bfcf4`|exact screened host:9/9 tests,30/30 graphs,5 static checks|

Drop-in `dev/glue/projection_glue.py`, frozen SHA256
`c763555f9908158879fb9bf65f766e03b650967487a8f97d325400613d5c92f9`.
API, integration and controls: `dev/glue/README.md`. Do not send the block as a candidate.
`gb_prepare(state,callbacks,deadline,recipes=...)` compares against the current best;
`gb_wrap_kernel` preserves the original kernel except shapes with a full-chain winner.
Empty recipes retain it. Keep native recipes uninstalled until actual H200 calibration.

Three launches replace O+boundary, pool/router+routing/norm, unpooler+boundary.
Full BF16 rounding boundaries, sigmoid biased top8, wrapped local ids and M<=8's
1024-column unpooler slice are preserved. Last CTA performs reductions/norm/route and
resets its own counter inside the timed kernel. No spinning or second reduce/add launch.
M1/8/16/32/128/256;300 phase/config cases,96 unique native specializations.
The control is exact frozen cand_r3_best812f5f36 projection/glue code, with its own
projection/Lt/PDL/prefix tuning, and ONE shared reset in the combined-chain control.
Weights are the loaded proj-host dense bank; missing norm/gates are synthetic calibration.
All three combinations (tail-only/front+tail/front-only) get cold1GiB/fresh3/20 pairs.
Optional failures reject/log. A per-stage result alone never enables dispatch.

Local sm120:300 actual configurations,900 fresh graph checks, all counters reset.
All96 cross-compile sm90. Hopper incumbent/Lt branches execute with portable native
Lt locally. Full fixture forces all18 paths:13/13 tests+12/12 graphs+11 static checks,
5.740s cached prepare. Final block's19 definitions/constants exactly match the fixture.
Lab host9/9+30/30+5 static,45.116s cached prepare; final summary<6KB. Fault injection
covers construction/launch/capture/oracle/coarse/final timing/budget/empty shortlist.
Shared sm120: every combined chain loses; closest M1 tail-only81.920->83.968us,ratio1.019.
H200 execution/timing, cold prepare and layer/neighbour-PDL effects remain unverified.

When the receipt lands, `dev/glue/read_receipt.py <receipt> --recipes <file>` validates
LAB_BEGIN's payload SHA and emits only phases from a positive combined-chain winner.
Retime that shortlist in the full decode layer; only its run proves a layer speedup.

Closed: stock CUTLASS lab45f58e7009c440eb, all144 recipes/key lose M128/256.
Best20-pair new/control ratios M256 qkv1.306,pool1.254,unpool1.082,o1.183;
M1281.322/1.244/1.142/1.186. Native recipes remain empty; family stays open.
Custom v2 job92e78027 loses1.21–1.95x; v1 job6b72087f is invalid. Do not resend either.
Old measured Lt block `dev/verify/projection_dropin.py` remains available, SHA97c87474…;
M128 qkv21.104->20.224us,unpool24.000->22.960us; full hashes in its README.
#45015 is serving-lane-owned. MoE is its lane's; no NVFP4 bank added by this block.
