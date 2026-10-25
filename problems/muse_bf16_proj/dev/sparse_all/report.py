"""Generate the matrix/load accounting and concrete P4 handoff from evidence."""
from pathlib import Path
import json
H=Path(__file__).resolve().parent
raw=json.loads((H/'sizes.json').read_text());p={r['matrix']:r for r in raw['rows'] if r['seed']==5353}
native=json.loads((H/'native_sizes.json').read_text());nf={(r['t'],r['matrix']):r for r in native['rows']}
def point(name):return p[name]['total_bytes']
def raw16(name):return p[name]['raw16_bytes']
def coded(name):return p[name]['raw12_bytes']+p[name]['cd_metadata_bytes'],p[name]['body_bytes']+p[name]['cd_metadata_bytes']
def nft(t,name):r=nf[t,name];return r['incumbent_bytes'],r['native_total_bytes']
def add(*pairs):return tuple(sum(r[i] for r in pairs) for i in (0,1))
def raw_point(name):return raw16(name),point(name)
def qkv(t,sliding):
    if t==1:return add(nft(1,'qkv3'),raw_point('gate'))
    if t==16 and sliding:return add(raw_point('qkv3'),raw_point('gate'))
    if t==32:return nft(32,'qkv_gate') if sliding else add(nft(32,'qkv3'),raw_point('gate'))
    return coded('qkv_gate') if t==1 else raw_point('qkv_gate')
def stages(t,sliding):
    return dict(qkv=qkv(t,sliding),o=nft(t,'o') if t in (1,8,16) else raw_point('o'),
        pool_router=coded('pool_router') if t==1 else raw_point('pool_router'),
        unpooler=raw_point('unpooler_slice') if t<=8 else nft(t,'unpooler') if t in (16,32) else raw_point('unpooler'))
loads=[]
for t in (1,8,16,32,128,256):
    a,b=stages(t,True),stages(t,False);r=dict(t=t,stages={})
    for name in a:
        control=(46*a[name][0]+16*b[name][0])/62
        sparse=(46*a[name][1]+16*b[name][1])/62
        # B materializes raw W for these library call sites. C reads the code
        # directly. This is a logical lower bound, not a physical HBM counter.
        materialize=t>=128 and (name!='qkv' or t==256)
        full='qkv_gate' if name=='qkv' else name
        b_traffic=sparse+2*raw16(full) if materialize else sparse
        r['stages'][name]=dict(control_bytes=control,code_bytes=sparse,saved_direct=control-sparse,saved_B=control-b_traffic,materializes_B=materialize)
    r['saved_direct']=sum(x['saved_direct'] for x in r['stages'].values());r['saved_B']=sum(x['saved_B'] for x in r['stages'].values());loads.append(r)
(H/'load_bytes.json').write_text(json.dumps(dict(provenance='sizes/native_sizes seed5353; one logical stream; 46 sliding +16 full layers averaged; no physical HBM measurement',loads=loads),indent=2)+'\n')
out=['# P10: sparse high-index plane on every dense BF16 stream','',
'Pinned implementation: S15 `60254883`; whole-layer integration belongs to P4. This directory changes no active layer candidate. Build with `python3 problems/muse_bf16_proj/dev/sparse_all/build.py`, then `build_lab.py`.', '',
'The transport stores eight low bits plus three fixed bits per weight (two high-index bits and sign), a presence/prefix pair per 1024 weights, and a sparse uint32 extension mask per occupied 32-weight lane. A circular rotation puts the four common high-byte indices in the fixed plane. Original escapes remain exact. Point readers add an indexed escape directory; native NF/CD readers reconstruct the original packed code and retain the existing decoder, MMA, split reduction and metadata.', '',
'The prior 11.451 figure was a particular NF bank, not a universal rate. The actual reference weights below have a different high-byte distribution. Router lo is the BF16 residual of the FP32 router after subtracting router hi; it is not an independently sampled matrix.', '',
'## Measured sizes', '',
'`sizes.json` uses the unmodified published `reference.generate_weights` including its expert RNG prefix, seeds 5353/61791/14357. Values below are seed5353. All bytes include a128-byte header. “Body” excludes point-escape/base metadata; “point” includes it. Savings are against one raw BF16 read, in **bytes**, at T=1/8/16/32/128/256. Gate is already included in full QKV; do not add it twice. Unpooler uses the first1024 output rows at T1/8.', '',
'| Matrix | N×K | Body bits/W | Point bits/W | Saved bytes T1 | T8 | T16 | T32 | T128 | T256 |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name in ('qkv_gate','qkv3','gate','o','pooler','unpooler','router_hi','router_lo','pool_router'):
    r=p[name];vals=[p['unpooler_slice']['saved_vs_bf16'] if name=='unpooler' and t<=8 else r['saved_vs_bf16'] for t in (1,8,16,32,128,256)]
    out.append(f'| {name} | {r["shape"][0]}×{r["shape"][1]} | {r["body_bits"]:.5f} | {r["bits_per_weight"]:.5f} | '+' | '.join(f'{x:,}' for x in vals)+' |')
out+=['','The short unpooler bank is `1024×4096`: '+f'{p["unpooler_slice"]["body_bits"]:.5f} body / {p["unpooler_slice"]["bits_per_weight"]:.5f} point bits/W. '+
'Point-format sizes measure the complete bank. M1 CD only streams the body plus its original scales/offsets/escapes; its unused point index is not counted as moved bytes. These are lossless BF16 banks, not a new numerical quantization.', '',
'Native NF formats have different tables and escape padding; their measured **total** rates and savings versus the existing12-bit stream are:', '',
'| T | Role | Body bits/W | Including NF metadata | Saved bytes |', '|---:|---|---:|---:|---:|']
for r in native['rows']:out.append(f'| {r["t"]} | {r["matrix"]} | {r["body_bits"]:.5f} | {r["native_bits"]:.5f} | {r["saved"]:,} |')
out+=['', '## Per-layer traffic accounting', '',
'These are minimum logical weight bytes, not profiler counters. Repeated CTA reads, partial outputs, cache-line amplification and activations are excluded. Values average the 46 sliding and 16 full layers; `load_bytes.json` retains exact arithmetic. QKV follows the current gate split: T1 both types; T16 sliding; T32 full. T1 pool/router keeps the coded CD reader, T32 keeps bal2. Table entries are MB (10^6 bytes) saved versus the current S15 stream, so prior12-bit compression is already credited to the control.', '',
'| T | QKV incl. gate | O | Fused pool/router | Unpooler | Total direct | Total B (timed restore at library sites) |',
'|---:|---:|---:|---:|---:|---:|---:|']
for r in loads:
    out.append(f'| {r["t"]} | '+' | '.join(f'{r["stages"][k]["saved_direct"]/1e6:.3f}' for k in ('qkv','o','pool_router','unpooler'))+f' | {r["saved_direct"]/1e6:.3f} | {r["saved_B"]/1e6:.3f} |')
out+=['',
'For a library arm B, timed weight traffic is bank-read + full BF16 scratch-write + full BF16 scratch-read. Its bank footprint savings therefore do **not** imply traffic savings. Direct C decodes inside our VDSK GEMM: BT128/BW64/BK64/NS3, four warps, activation TMA, S3 for K8192 / S2 for K4096 / S1 for K1024. This is route B only where opaque library kernels block injecting a reader. The T128 QKV B arm retains its current VDSK NS8/S3 structure.', '',
'## Arms and integration points', '',
'One payload contains 57 role/shape comparisons: QKV, O, gate, pooler, unpooler, router hi, router lo, fused pool_router and summed router at each T1/8/16/32/128/256, plus active QKV3 at T1/16/32. Each has independent A:A captures and B; 17 large-token sites additionally compare direct C. Eight activation seeds, eight alternating cold replays per seed, dense M1 inputs plus separate one-hot probes. Unpooler T1/8 maps to the short bank. Gate T128/256 is diagnostic because the active layer computes those rows in full QKV.', '',
'| Existing call site | Sparse B reader | What P4 must preserve |',
'|---|---|---|',
'| M1 full QKV and pool_router CD | reconstruct original int4 ring inside CD V3/PIPE2; same FP32 escapes | M1 remains coded; same split counts22/11 |',
'| Selected NF O1/8/16 and QKV3/QKV32 | native word reader before existing PRMT decode/MMA; CP staging stays asynchronous | original table origins, exact escape metadata, cluster reducer, PDL |',
'| Balanced QKV8/16, QKV3-16; pool8/16 | exact point reader inside existing GB dot loop | GB split counts and partial ABI |',
'| pool_router32 | exact point reader in existing bal2 loop | S6/BK128/NS2 and fused N4608 pool+hi+lo |',
'| gate split | exact point reader inside gate GEMV before gdc_wait | gate compute overlaps attention on the existing gate stream; last split waits before reading attention Y |',
'| unpool1/8 | point reader in existing GEMV | N1024 slice and existing padding/finalizer behavior |',
'| unpool16 | compact code staged by existing CP2 reader, then original D9 | same token guards and reductions |',
'| unpool32 | compact code expanded into original RS-WGMMA register-A words | same shared activation pipeline and FP32 partial output |',
'| O32 | exact point decode directly into current swizzled shared W ring; X TMA unchanged | SS-WGMMA, tile32×64, BK128/ST8; barrier expects X bytes only |',
'| QKV128 | exact point reader in current VDSK loop | activation TMA, S3, finalizer |',
'| Library-backed QKV256/O128/256/pool128/256/unpool128/256 | timed restore + identical library recipe; direct C comparison | do not promote based on footprint; short-Lt layout/ordinal and output precision stay explicit |', '',
'The fused pool_router path returns the original FP32 partial layout to the current route/norm consumer; separate pooler and router arms are diagnostics. Router hi+lo preserves the existing two-BF16 representation of the FP32 router. QKV3 and gate remain separate when split concurrency wins. O stays on the layer stream after gate/attention synchronization; unpool stays after experts on the layer stream. No proposed bank touches expert codes, KV caches, residual weights, norms or sigmoid precision.', '',
'## Verification and limits', '',
'`screen.py` runs `tools/local_eval.py` in the judge image, normal and forced, GPU0: nine TASK tests,30 shapes, three fresh inputs,25 static checks. Unchanged native CUDA reuses the exact admitted S15 proof; changed native transports are separately compiled by `compile_native.py` (nine sm90a units, no stack/spills), with the real NS8 T128 QKV Triton specialization compiled for Hopper. `compile_direct.py` compiles14 additional direct specializations. `test_readers.py` proves all original packed bytes through ordinary and CP-async readers in all three layouts, uniform and sparse occupancy (12 exact packed graph checks), plus two bit-exact CUDA point-reader checks for O and K8192 weights. `test_codec.py` checks every BF16 bit, signed zero and escapes for seven matrices; `test_paths.py` checks current paths and gate finalization; `test_direct.py` checks 16 direct role/shape cases over three seeds.', '',
'`test_hopper_glue.py` forces native cluster/MW/O32 plan preparation and launch arguments with arithmetic stand-ins (15 seed cases). `test_lab.py` rehearses the exact57+17-arm batch with real sparse decode and actual local direct kernels; Hopper-only cluster/WGMMA/large-shared launches use explicit arithmetic stand-ins. Validation synchronizes between arm replays because gate controls share partial/counter buffers and release dependents early. Timed samples use the cold read and completion events between arms. This fixes arm interference without adding work inside a measured projection. sm120 timings do not establish an H200 win. Native cluster/MW/O32 execution and overlap remain unverified until the queued lab.', '',
'## P4 decision rule', '',
'Send is independent of the reset result; queue this payload now behind existing prefill senders. P4 first reads the reset admission, then this lab. Require `END complete=57/57 direct=17`, every numerical/probe check, and an A:A envelope compatible with the claimed gain. Promote a cell only if median paired B/A (or C/A) beats1 by more than `max(0.02,2*max_AA_deviation)` across eight seeds and no seed regresses beyond its A:A envelope. Keep the admitted existing cell otherwise; delete losing development arms from the eventual integrated source. Rebase the codec onto whichever implementation the reset actually admits, rather than overriding a newer winner with this S15 control.', '',
'For gate/QKV3, compare the joint critical path with concurrent attention, not summed isolated times. For pool/router use fused consumer time, never add isolated pool+router wins. P4 must run paired whole-layer H200 tests on identical inputs and multiple weight/router seeds, keep current streams/fusions/PDL ordering, enforce exact escapes and FP32 router hi/lo, and admit only a measured layer gain. The final layer should retain only selected banks/readers; these lab A/A/B/C alternatives are not runtime fallbacks.', '',
'The NF-derived implementation retains the MIT notice naming Nathen Fernandes, [muse-bf16-proj@86aff75](https://github.com/NathenFernandes/muse-bf16-proj/tree/86aff75). The source and compressed payload include the complete copyright and permission notice. The previously reviewed vendor code performs no network/subprocess actions and writes only through torch JIT builds; this extension adds no such actions.', '',
'## Queue and evidence', '']
if (H/'queue.json').exists():
    q=json.loads((H/'queue.json').read_text());out.append(f'Prefill sender **{q["sender"]}**, priority{q["priority"]}, not-before **1792972800** (2026-10-26 09:00 KST). Wrapper `{q["wrapper_sha256"][:8]}`, payload `{q["payload_sha256"][:8]}`. No H200 job id or numbers exist until the sender runs. Read using `python3 tools/h200_lab.py read <job>`.')
else:out.append('Queue pending final local rehearsal; not-before1792972800, prefill, behind all existing senders.')
out+=['', 'Evidence: `sizes.json`, `native_sizes.json`, `load_bytes.json`, `manifest.json`, `reader_validation.json`, `codec_validation.json`, `path_validation.json`, `direct_validation.json`, `native_compile.json`, `direct_compile.json`, `hopper_glue_validation.json`, `lab_validation.json` and `admission.json`. Generated candidates, wrappers, native objects and logs are ignored; scripts regenerate them.']
(H/'README.md').write_text('\n'.join(out)+'\n')
