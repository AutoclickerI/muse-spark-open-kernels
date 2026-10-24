# M128 fused producer scale bank

Beat paired NT1-recoded at M128,283.6us in58ae2e2c, towards the~106us floor.
`build.py` freezes the shared repaired producer into raw/recoded controls and
copies the actual NT1 reader/packer from `cand_r3_v9.py`. It leaves the shared
producer, scheduler and W2 epilogue untouched. Current send: **def9bff54e82c1b0**;
judge **471f6d6216804dda** has landed: all numeric gates pass, but recoded cfg3/4/5
take 1649.96/1738.80/1851.23 us versus NT1-recoded 291.70 us. Recoding saves only
about1%; the serialized W2 combine masks it. No variant is promoted. The verified
distributed-finalizer port and shared-A reuse experiment are in `../one_read/`.

The recoded bank has the same9 bytes/block: eight packed E2M1 bytes and one
scale byte. For finite nonzero scale s, byte=(exponent+21)*8+mantissa encodes
|s|=2^(exponent)*(1+mantissa/8). Scale sign is XORed into both FP4 nibble signs;
zero scales zero the corresponding codes. Inside the producer,
`0x7000+(byte<<4)` is exactly the BF16 bits of |s|*2^118, including E4M3
subnormals. The existing bit-plane fragments represent E2M1*2^-126; BF16x2
multiplication and the existing global-scale factor256 restore the original
weights exactly. Four `__byte_perm` operations create the replicated scale
words, replacing four shared-table reads. No lookup initialization or E4M3
conversion remains in the recoded timed producer. Only packed codes/scales
survive prepare; the independent FP32 oracle is scratch for lab gates.

The lab uses cfg3(adaptive K64),cfg4(adaptive K128),cfg5(fixed NT128 K64), grid528,
with the K64 register-lifetime repair intact. Each of12 public-router weight
seeds has three fresh real-route inputs. Every implementation receives identical
x/IDs/route weights; six rotated/reversed replay orders flush128MiB first. NT1
appears twice per order for A:A noise. The per-fixture median and seed means are
reported. Gates check finite output,FP32 L2<=.01,row<=.05 before and after timing.
Separate gate/up global scales are covered by the standalone local cell gate;
the NT1 comparison uses the board's common W13 global scale.

Verified locally:36 fixtures,216 forced producer graphs; all4064 finite E4M3/FP4
value combinations exact. Wrapper5 official tests,6 scored graph shapes and6
static checks pass. Native12 kernels have no spills/C7513;20 K64 paths have
disjoint register banks. SM120 executes immediate MMA with the same Python and
pointer ABI and Hopper B128 addressing; this cannot establish H200 speed or
asynchronous numerical correctness. `agreement.json`, `native.json`,
`wrapper_screen.json` and `manifest.json` hold evidence/hashes.

Reproduce:

```bash
python3 problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/build.py
python3 tools/h200_lab.py build prof/lab/moe_m128_scale_bank.py --host muse_moe_w4a16 -o problems/muse_moe_w4a16/lab/lab_moe_m128_scale_bank.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/check.py
tools/img python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/native.py
python3 problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/build_dropin.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/check_dropin.py
```

`dropin.py` contains only the recoded producer, without raw controls or lab
dependencies. `dropin_manifest.json` gives prepare/call symbols and the two
M128 best7 rows. The standalone cell passes21 graph gates with signed, zero and
subnormal scales, separate gate/up globals and real/empty/single/hot/balanced
route transitions (L2<=0.004103,row<=0.004601). H200 latency rejects this old
serialized-finalizer cell; its manifest remains disabled. Current performance
work and the replacement drop-in are in `../one_read/`. NT1-recoded stays active
until a paired H200 winner passes full-layer admission.
The earlier K64 repair was batched into97be11d4631b44d6 and passes all12 seeds,
R0 and HOT128 with finite scratch; numeric repair is H200-confirmed.
