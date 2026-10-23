"""Generate reviewable local evidence tables, retaining source hashes and failed experiments."""
from pathlib import Path
import json,math,hashlib,datetime
p=Path(__file__).resolve().parents[1];dev=p/'dev'
rows=[]
for name in ['stage_comparison','stage_comparison_extra','stage_comparison_occ2']:
 f=dev/(name+'.json')
 if f.exists():rows+=json.loads(f.read_text())['rows']
by={}
for r in rows:
 name=r['candidate']
 if name=='cand_chain_occ2' and hashlib.sha256((p/(name+'.py')).read_bytes()).hexdigest()!=r['sha256']:
  name='forced_occ2_retired'
 by.setdefault(name,{})[r['case']]=r
base=by['cand_v3'];loads=['b1','b16','b32','v1','v16','v32']
lines=['# Structural decode round: local evidence','',
'All times here are measured on the local RTX PRO 6000 (sm_120), inside the judge image. They are not H200 estimates. No new remote GPU run or commit was made. The shipped submission remains v3. `cand_persistent.py` is a superseded prototype, not in the proposed check set.',
'',
'`structural_bench.py` uses common generated weights and seed 77, cold-L2 graph replay (1 GiB flush), 20 replays/stage, and subtracts the measured replay fixed cost. Baseline and candidates run serially on GPU 0. Attention uses the portable implementation locally; FA3 is Hopper-only. Cross-run comparisons and small gaps remain screening evidence. Stage graphs break cross-stage PDL overlap; their times must not be added to predict whole-layer time.',
'',
'## Full-layer comparison with v3 and every round-2 candidate','',
'Ratios are v3 time / candidate time, first aggregating sliding/full with weights 46/16, then taking the six-load geometric mean. Above 1 is faster. These are **local**, serial comparisons, without confidence intervals. JSON files retain exact source hashes.',
'',
'| Candidate | Overall vs v3 | D1 | D16 | D32 | V1 | V16 | V32 |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
ratios={}
for name,cases in by.items():
 if len(cases)<12:continue
 vals=[]
 for l in loads:
  a=sum(base[l+k]['us']['whole']*w for k,w in [('s',46),('f',16)])
  b=sum(cases[l+k]['us']['whole']*w for k,w in [('s',46),('f',16)])
  vals.append(a/b)
 ratios[name]=[math.prod(vals)**(1/6)]+vals
 lines.append('| '+name+' | '+' | '.join(f'{v:.3f}x' for v in ratios[name])+' |')
lines+=['','## Per-stage graph times','',
'Times in microseconds. Prefix groups O projection + first residual/norm + pooler/router + routing/pre-expert norm. Suffix groups unpooler + final residual/norm. The fused candidates use one kernel for each island; the epilogue candidate folds boundary arithmetic into O and pooler/router instead. All 12 cases for all candidates are in [stage_comparison.json](stage_comparison.json), [extra](stage_comparison_extra.json), and [adaptive occupancy](stage_comparison_occ2.json).',
'',
'| Candidate | Case | QKV | Attention | Prefix | Experts | Suffix | Whole |',
'|---|---|---:|---:|---:|---:|---:|---:|']
for name in ['cand_v3','cand_moe4_attn','cand_native_glue','cand_chain','cand_chain6','cand_epilogue','cand_chain_occ2','cand_chain_nopf']:
 for case in ['b1s','b16s','b32s','v1s','v16s','v32s']:
  if name not in by or case not in by[name]:continue
  r=by[name][case]
  lines.append('| '+name+' | '+case+' | '+' | '.join(f'{r["us"][s]:.1f}' for s in ['qkv','attention','prefix','experts','suffix','whole'])+' |')
lines+=['','## Exact-source validation','',
'Each row is from `tools/img python tools/local_eval.py`: all 13 official local test cases, followed by all 12 shape graphs on three fresh input sets. The graph checks use new inputs in captured buffers. Passing local tests does not validate the Hopper FA3 path or the judge\'s router seed.',
'',
'| Candidate | Tests | Graph cases | Prepare seconds (cache-dependent) | SHA-256 |',
'|---|---:|---:|---:|---|']
validation={}
for short in ['chain','chain6','chain_nopf','chain_occ2','epilogue','layer_sweep']:
 f=dev/f'local_{short}.json'
 if not f.exists():continue
 d=json.loads(f.read_text());source=p/f'cand_{short}.py';h=hashlib.sha256(source.read_bytes()).hexdigest()
 same=h==d['sha256'];ok=same and len(d.get('tests',[]))==13 and all(r['ok'] for r in d['tests']) and len(d.get('shapes',[]))==12 and all(r['ok'] for r in d['shapes'])
 validation[short]={'ok':ok,'hash':h}
 lines.append(f'| cand_{short} | {sum(r["ok"] for r in d.get("tests",[]))}/13 | {sum(r["ok"] for r in d.get("shapes",[]))}/12 | {d["prepare_s"]:.1f} | `{h}`'+(' |' if same else ' **SOURCE CHANGED** |'))
lines+=['','## Actual captured PDL chain','',
'`graph_inspect.py` queries `cudaGraphGetEdges` through CUDA bindings and saves edge metadata, DOT graphs and GPU timelines. For T16/T32 decode there are **six compute kernels plus one zero-fill kernel**, with **six programmatic edges** (type 1, outgoing port 1). Batch-1 decode/verify uses the portable three-kernel attention path locally: seven compute kernels plus the zero-fill, seven programmatic edges. The GPU trace shows overlap, e.g. prefix starting before attention combine completes. Kernel durations include waits; earlier launch alone is not a speedup. See [edge metadata](graph_16_1_edges.json), [timeline](graph_16_1.json), [counts](graph_inspect.log).',
'',
'Above T32, cuBLAS projection and existing boundary dispatch are retained. Local verify-16/32 has 15 GPU activities, not a six-kernel fused path. Hopper verify retains FA3. This round does not claim a single kernel across the whole layer or across all token counts.',
'',
'## Decisions, failures and static Hopper checks','',
'- The two persistent islands fuse O/post-attention norm/residual/pre-FFN norm/pooler/router/top-8/pre-expert norm, and unpooler/post-FFN norm/residual/next-input norm. Expert computation is one persistent NVFP4 kernel. Grid sizes are bounded by queried occupancy; token-local norm chunks are required to fit the resident grid. The normal templates have no ptxas spills.',
'- Every custom attention and MoE kernel now launches with PDL, releases dependents in its independent prologue, and waits before accessing prior-stage outputs. Earlier v3 PDL existed only in its Triton path; merely turning on a flag would not close those gaps.',
'- Immutable O weights (16 MiB) are prefetched before the prefix waits; 8 MiB sliced or a bounded 16 MiB full unpooler prefix is prefetched before the suffix waits. Hopper uses `cp.async.bulk.prefetch.L2.global`; sm_120 uses `prefetch.global.L2`. Prefetch is a performance hint. Pooler/router already prefetch their own known weight slices before waiting in the epilogue chain. No expert weight is speculatively chosen or prefetched before routing. Prefetching all dense banks together was rejected on capacity/order grounds: later banks would displace O; there is no measured benefit to claim.',
'- Initial single-CTA residual norms cost ~16 us locally; their routing phase cost ~11 us. Token-local multi-CTA statistics and one-warp top-8 fixed correctness and reduced those overheads, but whole-island fusion still regresses locally. `profile_persistent_initial.log` and `island_clock.log` retain the diagnosis; the suffix intermediate timestamp slots in the initial clock log were unpopulated and are not timings.',
'- Forced two-CTA residency at every tile size caused Hopper spill stores/loads (up to 688/580 bytes at NT4) and a large local T32 regression. That experiment is retired in `occ2_forced.py`, its JSON/log and `sm90a_occ2_forced/` compiler evidence. The revised `cand_chain_occ2` only requests two-CTA launch bounds for NT1 (T<=8); NT2/4 keep one-CTA bounds. Runtime occupancy clips the actual grid. Its final exact-source checks pass; the validation table above and `sm90a_occ2/` describe the revised variant.',
'- The narrower epilogue alternative publishes four per-tile sufficient statistics in O, uses the last CTA to compute row norms, and applies the residual/pre-FFN norm in pooler/router loads, writing stream1 once. It preserves BF16 activation rounding. It removes one boundary launch and the materialized hidden buffer, but has extra traffic and last-CTA reduction cost.',
'- H200 MoE v6 receipt `eefecf1eeb724feb` (1.4916x standalone) superseded v4 while this round ran. `PROFILE.md`, `research/moe_decode.md` and `research/hw_lowlevel.md` support keeping it as a separate candidate: it loses locally but wins on H200. Thus the v4 structural control remains available. All large-T improvements in the v6 candidates currently come from MoE, not new dense fusion.',
'- `sm90a/manifest.json` and `sm90a/epilogue_manifest.json` record successful static compilation of the new CUDA templates and Triton epilogues. This is compile evidence only. FA3 execution, H200 cold-L2 gains, true overlap savings, compilation against a cold judge cache, and alternate router seeds remain unverified.',
'- The early epilogue compile failed because an inline-assembly `pack` argument was missing; it was corrected, without changing checks. The first prototype misdeclared device `local_start` as int64; it is int32 and was corrected. Failed logs are retained.',
'- The official upstream sync ran in the existing isolated mirror under this dev directory. TASK, official AGENTS, reference, checks, baseline, NVFP4 helper and llms.txt compared unchanged. Other problem files, tools and serving were not edited by this round.',
'',
'## Reproduce','',
'```sh',
'CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_decode_layer/cand_chain6.py --json problems/muse_decode_layer/dev/local_chain6.json',
'CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/structural_bench.py cand_v3 cand_moe4_attn cand_chain6 cand_epilogue --json problems/muse_decode_layer/dev/recheck_stages.json',
'CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/compile_structural.py',
'CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/compile_epilogue.py',
'CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/graph_inspect.py',
'```','',
'PDL semantics were checked against [NVIDIA\'s programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); bulk-prefetch requirements against the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/). The design does not depend on concurrent execution for correctness. External megakernel results in SOTA are motivation, not measured gains for this layer.']
(dev/'STRUCTURAL.md').write_text('\n'.join(lines)+'\n')
(dev/'structural_summary.json').write_text(json.dumps({'ratios':ratios,'validation':validation},indent=2))
print('report written;',len(rows),'stage rows;',validation)
