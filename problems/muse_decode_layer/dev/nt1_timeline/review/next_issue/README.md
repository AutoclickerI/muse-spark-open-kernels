# P3a instruction cuts on the measured composition

Parent: exact c49 composition,SHARD2 at M1,PAIR at M8,PAIR+BLOCK at M16/M32.
`../combo/results.json` audits16 fresh seeds and64 A:A pairs; native/official TASK passes.
COMBO/S15=.997261/.988757/.970993/.979561; direct COMBO/PAIR at M16/M32=.995504/.993125.
P4 receives the exact combined S15 patch in `../../point_nominations.json` with ratios,SE and seeds.
Only P4's held-out six-load R confirms promotion.

## What CANCEL added

`audit_cost.py` compiles the actual S15,CANCEL and candidate CUDA with sm90a,-O3,-lineinfo.
`sass_cost.json` keeps both flags69/95 resources and all opcode counts. The flags69 comparison:

| Build | Main-kernel instructions / text bytes | Registers / spills | LDG / LDS / FSETP | FP64 DMUL / DFMA |
|---|---|---|---|---|
| S15 |3168 /50688 |125 /0 |77 /143 /15 |0 /0 |
| Original CANCEL |4072 /65152 |125 /0 |100 /159 /76 |64 /169 |
| FAST on S15 |3264 /52224 |125 /0 |84 /159 /23 |0 /0 |

CANCEL's finalizer maps each sorted pair back to an expert,loads epsilon and sign,checks zero/range,
then either runs the normal FP32 path or a duplicated FP64 exceptional path. It adds23 static global
loads and16 shared loads; the shared delta matches2 expert-map gathers at each of8 unrolled ranks.
The full static count includes cold paths and library calls. It does not prove that ordinary scales
execute FP64. The unchanged125-register footprint and zero spills show no added occupancy/spill cost.
The measured1–2% loss cannot be attributed uniquely from these static counts.

FAST uses the same real-number identity with bounded FP32 metadata. Let q=a/sqrt(eps),a=256*gs2[e].

- |q|>=1: mul=1,bias=4096/q²,num=64*sign(q).
- |q|<1: mul=q²,bias=4096,num=64*q.
- Final factor=num*rsqrt(fma(sum,mul,bias));sum=0 returns0.

The sign is in num; zero global scale gives num=0. Metadata is one aligned FP32 float4 per expert,
including the original FP32 global scale; no NVFP4 value is widened or predecoded.
There is no runtime FP64 or sign/exponent branch. W13/SwiGLU is unchanged.
`fast_extremes.json`:224 native helper checks,zero/signed/tiny/huge scales and norm sums down to1e-40,
worst relative9.33e-8;16 captured complete-MoE zero/negative/1e-10/1e10 scale TASK checks pass.
The new PAIR+BLOCK FAST image uses124 registers and zero spills. H200 receipt d7fb2d8bddae4b8c is complete;the verdict is below.

## Next integer/address cut

INDEX moves row-valid/pass branches outside the8 B-cache fills and uses a bounded32bit vector index.
At M<=32,the largest sorted position is255; the activation/intermediate offsets fit comfortably in32bits.
`audit_cost.py` marks the actual old/new fill bodies and keeps every output observable:

| Marked fill | Old → INDEX integer/address/control instructions | Vector loads |
|---|---|---|
| W13 |31→13 |16→16 |
| W2 |26→8 |16→16 |
| Runtime-pass full static set |63→20 |32→32 across the two branches |

The finite-pass static-set reduction is18 per segment. There are16 requested16-weight groups per warp
tile,so the model saves18/(16*L)=1.125/L instructions per16 weights at segment length L. This is an
amortized setup cut; the per-tile decoder is unchanged. Static sets include empty branches; they are
not dynamic instruction/stall counts. The new PAIR+BLOCK image uses122 registers,zero spills.
The earlier17 integer/address instructions per16 in the core decode/scale body remain.
`zero_rotation_search` exhaustively rejects only the restricted same-mask3-pure-slots/one-LOP3
layout with a zero rotation in both words; it does not close other packed decoders.

FLUSH1 is independent: PAIR calls flush_vec_norm twice consecutively at segment transitions.
The first call resets nv0; the second only adds zeros. Remove that second call,retain every required
flush/publication/fence/acquire. The count build falls3352→3296 instructions,ATOMG7→6,125regs/zero spills.
This also leaves other norm ownership/scheduling open. The measured verdict below decides these exact cuts.

## Arms and validation

`manifest.json` pins three independent candidates and exact S15 patches:FAST,INDEX,FLUSH1.
Each has3 per-M variants; BASE CUDA is byte-identical to c49's admitted variants.
`component_validation.json`:90 TASK/current-input/capture checks per variant,810 total,including
empty/all8/zero routes,M9/M24,nonzero expert starts and changed replay inputs.
Normal/forced `tools/local_eval.py` reports are in `screens.json`;only exact-hash successful reports count.
Changed MoE arithmetic runs natively on GPU0; local Hopper-only attention glue uses explicit stand-ins.

`build_lab.py` makes `prof/lab/moe_p3a_next_issue.py`:BASE/S15/FAST/INDEX/FLUSH1,16 fresh seeds/M,
64 independent BASE:A pairs,each arm paired against BASE and S15,TASK before/after,read-flushed AB/BA.
Weights375353+7919*s,routes379000+M+97*s. The bounded receipt retains448 seed ratios and64 A:A times.
The wrapper is975473 bytes,below the server1MB source limit. Local timing is never H200 evidence.
Queue/admission records pin the exact screened send on the MoE host.

Reproduce:

```sh
python3 problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/build.py
python3 problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/audit_cost.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/check_components.py FAST0 FAST1 FAST2 INDEX0 INDEX1 INDEX2 FLUSH10 FLUSH11 FLUSH12 --manifest problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/components_manifest.json --output problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/component_validation.json
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/check_fast.py
python3 problems/muse_decode_layer/dev/nt1_timeline/review/run_screens.py --manifest problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/screen_manifest.json --output problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/screens.json
python3 problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/review/run_lab_local.py --payload prof/lab/moe_p3a_next_issue.py --output problems/muse_decode_layer/dev/nt1_timeline/review/next_issue/lab_local_validation.json
```

Sent check**`d7fb2d8bddae4b8c`**,sender1971066,wrapper`76a7f192a080d0b6`,on MoE. All6 normal/forced13TASK/12shape/28static screens pass; exact packed16seed/64AA local lab passes5919characters/94.7s. `admission.json` and `queue.json` pin the send. H200 receipt complete;verdict below.

Instruction-cut receipt d7fb2d8bddae4b8c is complete:all16seeds/TASK gates pass. FAST/INDEX direct M1 point wins go to P4;all3 recipes lose directly to BASE at M8/16/32. Exact per-M ratios,CI,seeds and patches are in `../../point_nominations.json`;no standalone small-cut rerun. Structural depth/phase batch48dc8b04902b40b1 is complete;see `../../structural/README.md`.
