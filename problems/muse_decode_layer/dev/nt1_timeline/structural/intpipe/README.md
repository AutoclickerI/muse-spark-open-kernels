# Integer issue on exact S19

Base: frozen S19 expanded `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`.
`build.original()` always reconstructs the expanded base from the frozen tracked packed candidate.
The cells are extracted from that exact source, including pdl_M, FAST+INDEX+SHARD2 M1 and
PAIR M8 / PAIR+BLOCK M16/32. Patches apply to `dev/s19/generated/s19.py`, reconstructed from the
frozen packed candidate by its own checksummed wrapper. `ship.py` uses that same wrapper to ship
single files below the size limit. No other S19 kernel or prepare choice changes.

## Claims tested before GPU work

`premises.py` checks all 4,064 finite signed E2M1 x E4M3 products (including scale subnormals):
FP4*scale/256 is exactly representable in both FP16 and BF16. This fact does not itself give an
integer-free decoder. The usual affine INT4 magic formula produces 2.5/3/3.5 for E2M1 codes5/6/7,
which require3/4/6. An exhaustive 1,838,032-case search over three magnitude-bit positions and every
constant positive BF16 base finds exactly the incumbent zero-bias tiny-BF16 embedding (bits6/7/8).
No nonzero-bias single-affine form is established. This rejects that bounded form only; multi-piece
floating decoders remain open. No A arm was sent or called a measured loss.

The requested direct one-LOP3/PRMT zero-shift extraction also has no implementation here. With direct constant-index
byte permutation to tiny-BF16 registers, only source byte bits0/6/7 can reach required magnitude/sign
positions:24 available independent source bits versus64 in16 independently variable codes. The
same-mask four-disjoint-rotation search also has no solution. These are bounded bit-selection/layout
results, not a lower bound on all lossless recodings, arithmetic decoders or dynamic-selector lookup methods. No B arm
was sent or called a measured loss. Existing packed-code LUT implementations already lost on H200.

S19 already uses bulk TMA copies and affine weight-tile addresses; merely introducing TMA is not new.
The remaining address/control implementations are measured independently in one batch.

## Arms and local sizing

AFFINE preserves the M8+ B-fill row-valid/pass branches and all16 vector reads per segment. A 64-bit
row base feeds explicit constant-offset vector loads with unchanged .cg/.nc policies. M1 remains
S19's INDEX path, an identical control alias, excluded from nominations.

PRIVATE transfers the TMA cursor from shared Route storage to owner registers once, after initial
warp0 copies have published. Warp7 lane0 uses its private cursor throughout the steady loop. Initial
copy overlap, all math, store/fence/publish/acquire and the grid barrier remain. It adds no warp or slot.

Exact sm90a SASS and CUDA13 occupancy model: all arms retain two CTAs at two raw slots. BASE registers
M1/M8/M16+ =117/126/126, AFFINE117/126/126, PRIVATE121/119/119. PRIVATE has8/16/16B call stack,
zero spill loads/stores; BASE/AFFINE have zero stack/spills. Shared bytes for every arm M1/8/16/32:
83904/84176/84496/85136. H200 runtime resource rows must match before a measured verdict.

Full-image INT sites M1/M8/M16+: BASE986/1799/1811; AFFINE986/1754/1769;
PRIVATE989/1633/1647. PRIVATE M8+ uniform-class sites rise66→292, shared LDS/STS sites fall227→213
(M8) or226→212 (M16+). Full-image counts include alternative/exception paths and are not executed
instructions per weight. All arms retain exactly the same457-instruction math body:17.0625 INT,1 PRMT,
8 packed BF16 multiply,2 HMMA and.5625 weight/scale LDS per16 requested weights per lane.
The exact457-site marker has273 INT,16 PRMT,128 HMUL2,32 HMMA and8 LDS;the initial scale LDS
is outside. This corrects the earlier416/32/9 bucket label,which mixed those two scopes.
`sass_model.json` gives per-M logical-repeat factors and empirical same-op issue envelopes. INT is
largest in that modeled body at all M; actual hardware saturation is unmeasured.

## INDEX regression audit

The d7fb2d8b receipt's marked B-fill cuts31→13 (W13) and26→8 (W2) integer/address/control sites per
segment. Whole-kernel compilation also adds2 VOTEU,33 LEA,53 UIMAD and82 UIADD3 sites while deleting
other instructions and changing uniform control throughout the image. The recount includes predicated @UP
instructions that the previous opcode parser omitted (UIADD3 delta82, previously63). Loads, spills and residency
do not increase. These added sites expose the tradeoff; static counts cannot identify the unique
cause of a1% runtime loss. AFFINE retains the old control structure rather than reusing INDEX's hoist.
M1's already-integrated INDEX remains in the S19 base.

## Admission and one-batch decision

Use static first, changed-source sm90a compilation and one import transport smoke via local_eval.
All324 native TASK/current-input/capture checks pass on GPU0, including empty/zero/all-local routes,
nonzero local starts and changed inputs under capture. Changed kernels execute directly on sm120;
no stand-in is used. The exact packed lab validates16 seeds,128 paired ratios and64 A:A pairs before
sending. H200 weights975353+7919*s; routes979000+M+97*s. Timing compares the exact S19 cells, independent
AFFINE/PRIVATE arms, read-flushed AB/BA, TASK before and after. No full-layer R is claimed.

Every valid per-shape point winner, with SE/CI/seeds and exact S19 patch, goes through
`point_nominations.json` to P4; it confirms held-out six-load R. A failed control or implementation
is unmeasured. A:A and per-shape CI describe uncertainty and do not veto nomination.

## H200 verdict

**`6f5afe0c273e4bbf`**, sender1006438, wrapper01a7a9c1aeed4525, completes16 fresh seeds at M1/8/16/32,128 paired ratios and64 A:A. Exact frozen S19 expanded62ea5bf4 is the BASE, including FAST+INDEX+SHARD2 M1, PAIR M8, PAIR+BLOCK M16/32 and pdl_M. Official host and all native TASK gates pass, no failures; max total/row .002456/.002491. Packet checksum, submitted source/cells, all12 runtime resource rows and seed completeness are verified. Candidate/control time; descriptive95% Student-t15 CI:

| Recipe | /S19 BASE [95% CI], M1; M8; M16; M32 | Mechanism / decision | Leaves open | Receipt |
|---|---|---|---|---|
| AFFINE | 0.997260 [0.994542, 0.999978]; 1.000827 [0.996470, 1.005183]; 1.002168 [0.998112, 1.006224]; 0.997026 [0.994998, 0.999054] | Keeps row-valid/pass control; one 64-bit activation row base plus constant-offset vector loads, same16 .cg/.nc reads. M1 unchanged control alias, excluded; M32 point nominee. | Larger decoder cuts and other address/control schedules; no reduction of core decode instructions. | `6f5afe0c273e4bbf` |
| PRIVATE | 0.997059 [0.994298, 0.999820]; 1.008689 [1.003961, 1.013417]; 1.011149 [1.005550, 1.016748]; 1.015676 [1.011441, 1.019912] | Transfers steady TMA cursor from shared Route to issuing-owner registers after original initial copies; all8 math warps and2 raw slots retained. M1 point nominee; M8+ loses despite fewer static INT/shared sites. | P4 whole-layer R for M1; other producer-state representations without the added uniform/call-stack tradeoff, arithmetic/dynamic-lookup decode. | `6f5afe0c273e4bbf` |

A:A mean ratios at M1/8/16/32 are1.004719/1.003344/1.001305/.999876, SE .001485/.001874/.002525/.001268; mean absolute .174/.416/.508/.348us, max .672/.736/2.064/.976. M1 has positive A:A drift and its identical AFFINE alias also shows .997260 [ .994542,.999978 ]; PRIVATE M1 is therefore a weak point nominee, not established implementation benefit. The G0 per-shape point rule still hands it to P4 with this alias diagnostic; no A:A/CI veto or standalone confirmation run.

Per-pipe sizing: exact old457-site marker is273 INT (17LEA+112SHF+144LOP3),16 PRMT,128 HMUL2,32 HMMA,8 LDS. The scale LDS sits outside that marker. This corrects the earlier416 decode/32 MMA/9 LDS label, which mixed scopes; neither the kernel nor old timings change. All new arms retain that core exactly. Normalizing per16 requested weights/lane gives17.0625 INT,1 PRMT,8 BF16 multiply,2 tensor instructions plus.5625 weight/scale LDS. PIPE16c18078's empirical same-op envelopes1.96/1.96/3.57/.658/.25 imply these modeled SM cycles per unique weight before **and after** both arms:

| M | INT | PRMT | BF16 arithmetic | Tensor | Weight/scale LDS | Largest modeled issue demand |
|---|---|---|---|---|---|---|
|1 /8|.5441|.0319|.1401|.1900|.1406|INT|
|16|.6534|.0383|.1682|.2281|.1689|INT|
|32|.8654|.0507|.2228|.3021|.2237|INT|

Factors1/1.20098/1.59052 are historical logical weight-request repeats, not measured HBM misses. Actual saturation, exclusive scheduler stalls and a complete whole-kernel cycles/weight model remain unmeasured; uniform throughput is not calibrated. Do not add the separate operational envelopes as independent physical ports. Address/control changes are separately sized below.

Full-image INT/uniform/FMA/tensor site counts M1;M8;M16+ are BASE986/73/267/32;1799/66/292/32;1811/66/292/32. AFFINE986/73/267/32;1754/66/292/32;1769/66/292/32. PRIVATE989/71/267/32;1633/292/292/32;1647/292/292/32. M8+ PRIVATE replaces14 shared LDS/STS sites while adding226 uniform sites; these image counts include alternative/exception paths, not executed per-weight work. Marked finite B-fill BASE→AFFINE per-segment INT counts are31→11 (W13) and26→10 (W2); every form retains16 vector reads. With segment length L tiles, their INT contribution is .06178/L→.02192/L (W13) and.05182/L→.01993/L (W2) SM cycles per requested weight, before historical repeat factors. Core INT/FMA/tensor costs stay unchanged. PRIVATE's added uniform throughput is uncalibrated, so a predictive whole-kernel cycles/weight total is unavailable. PRIVATE all-M stack allocation8/16/16/16B is real, withzero spill stores/loads; BASE/AFFINE zero local. Every arm actually keeps2CTAs. Registers M1/M8/M16+ BASE117/126/126, AFFINE117/126/126, PRIVATE121/119/119; dynamic shared83904/84176/84496/85136B, unchanged.

INDEX's earlier d7fb2d8b B-fill31→13/26→8 cut changes uniform control across the full kernel: adds2 VOTEU,33 LEA,53 UIMAD,82 UIADD3 sites while removing others. Loads/spills/residency do not grow. This identifies added work before reusing the idea, not a unique runtime cause of the loss. AFFINE preserves the prior control; it does not reuse INDEX's branch hoist at M8+.

A:all4,064 finite signed E2M1 x E4M3 products/256 are exactly representable in FP16 and BF16. The usual single-affine magic-nibble form is not an exact E2M1 decoder (codes5/6/7 give2.5/3/3.5 instead of3/4/6). Exhaustive1,838,032 positive-BF16 magnitude-bit/base embeddings find only the incumbent zero-bias bits6/7/8 form. No proposed nonzero-bias HFMA2/HADD2 implementation is established; no A arm was sent or measured. Leaves open multi-piece arithmetic, other recodings and mixed conversion forms; this bounded result does not close FMA decode.

B:the bounded direct constant-index LOP3/PRMT zero-shift selection family cannot carry64 independent input-code bits through24 available source positions; four same-mask disjoint rotations also have no solution. No new nibble-permutation arm was established or measured. Leaves open dynamic-selector PRMT tables and arithmetic/layout combinations; these searches are not a general decoder lower bound. S19 already issues weight copies with bulk TMA, so simply adding TMA is not a new C arm. `structural/intpipe/premises.json` retains exact scopes and counterexamples.

P4 receives PRIVATE M1 (.997059, SE .001295,16 seeds) and AFFINE M32 (.997026, SE .000952,16 seeds) in `point_nominations.json` and `structural/intpipe/P4_HANDOFF.json`. Exact patches apply to the checksum-verified expanded S19: PRIVATE `private.patch` e0127e8da0a8e75ba9ceb0e6c362cb73f33a2200cfb5fc85b9698c7afa0f3ee3; AFFINE `affine.patch`4ebd938516334d5a79a69d968e86c76275f8f9b8594bb13d7699ded8676108cd. P4 confirms held-out six-load R; no robust or additive gain is claimed.

Static-first checks, all changed native sm90a units and import transport smoke pass;324 native TASK/current-input/capture checks pass without stand-ins. Exact wrapped local16-seed packet passes128 pairs64AA,5794B/81.1s. A send preflight first rejected a missing top-level lab entry before quota. The final queue command continued after a stale fingerprint report while its wrapper rerun finished; expanded body was unchanged and final full admission now passes. The actual H200 packet is complete,25.5s including compile. No failed test is suppressed.

The core per-weight decode lever remains uncut. Next by ceiling is a genuinely cheaper exact dynamic-selector/arithmetic decoder, proved on all4,064 products and counted against the unchanged S19 math body before another run. These small address nominees travel in P4's next composition; no standalone sub1% confirmation.

