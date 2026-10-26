"""Rewrite only our projection/P12 measurement block; optionally stage that hunk."""
from pathlib import Path
import json,subprocess,difflib,tempfile,sys
H=Path(__file__).resolve().parent;R=H.parents[3];F=R/'docs/FINDINGS.md';A=R/'problems/muse_bf16_proj/dev/sparse_all'
tag='### P10/P12 reset measurements';end='<!-- P10/P12 measurements end -->'
rows=[]
def cell(x):return str(x).replace('|','/').replace('\n',' ')
gap=json.loads((H/'leader_gap.json').read_text())
for r in gap['ranking']:
    if r['passed']:
        ratio=f"SGLang/ours {r['our_speedup']:.5f} [{r['speedup_ci'][0]:.5f},{r['speedup_ci'][1]:.5f}]; ours/leader {r['time_ratio_to_component_leader']:.5f}; A:A unavailable"
        scope=f"{r['tests']} tests/{r['shapes']} shapes pass. Leader per-shape times unpublished; aggregate gap only; no layer promotion."
        if r['role']=='projection':scope+=' Includes board sum/cast/router-add tax; one-reducer/producer-only follow-up queued.'
    else:ratio='unmeasured: component correctness/setup failed';scope=r.get('error') or 'Inspect the receipt before rebuilding this adapter.'
    rows.append((f"P12 S15 {r['role']} component",ratio,'Frozen S15 path behind the board ABI; official paired SGLang control',scope,r['job']))
result=json.loads((A/'results.json').read_text())
for run in result['runs']:
    if not run.get('job'):continue
    if run['status']!='measured':
        rows.append((f"P10 {run['kind']} sparse batch",'unmeasured: failed or incomplete control/gate','No recipe loss or promotion inferred','; '.join(run.get('error_lines',[])) or run.get('host_error') or 'Inspect omitted/incomplete lab output',run['job']))
        continue
    for r in run['rows']:
        if run['kind']=='pool':
            mechanism='NF weights-on-M warp MMA, fused N4608 route/norm; '+('parallel sparse expansion' if r['arm'].endswith('vec') else 'scalar sparse expansion' if r['arm'].endswith('sp') else 'original12-bit reader')
            if 's4' in r['arm']:mechanism+='; S4 DSM'
        elif r['arm']=='C':mechanism='Direct VDSK, timed sparse decode inside GEMM'
        elif r['t']>=128 and (r['t'],r['role'])!=(128,'qkv_gate'):mechanism='Timed restore plus current library; full scratch traffic included'
        elif r['role']=='gate':mechanism='Scalar point decoder in gate GEMV; original last-split wait'
        elif (r['t'],r['role']) in ((1,'qkv_gate'),(1,'pool_router')):mechanism='Sparse words reconstructed in current CD ring'
        elif (r['t'],r['role']) in ((1,'o'),(8,'o'),(16,'o'),(1,'qkv3'),(32,'qkv3'),(32,'qkv_gate'),(16,'unpooler'),(32,'unpooler')):mechanism='Sparse words reconstructed before existing NF decoder/MMA'
        elif (r['t'],r['role'])==(32,'o'):mechanism='Point decode into current SS-WGMMA shared W ring'
        else:mechanism='Scalar point decoder in current balanced/bal2/GEMV loop'
        decision=r['classification']+'; closes only this reader/structure combination, not lossless coding'
        if r['followup_candidate']:decision='P4 whole-layer candidate; require paired layer gain and preserve fusions/concurrency'
        if run['kind']=='pool' and r['t']==1:decision+='; this M1 batch times one-hot activations only'
        if r['role'] in ('pooler','router_hi','router_lo','router_sum') or r['role']=='gate' and r['t']>=128:decision+='; standalone diagnostic'
        rows.append((f"P10 {run['kind']} T{r['t']} {r['role']} {r['arm']}",f"{r['ratio']:.5f} [{r['ratio_min']:.5f},{r['ratio_max']:.5f}]; AA max {r['aa_max']:.5f}",mechanism,decision,run['job']))
block=tag+'\n\nRecipe ratios are candidate/control (lower is faster), except the explicitly labelled component speedups. '
block+='Full shape/seed evidence and code pins: `problems/muse_bf16_proj/dev/sparse_all/results.json` and `problems/muse_decode_layer/dev/p12_components/leader_gap.json`.\n\n'
block+='| Recipe | Ratio and control spread | Mechanism | What remains open / decision | Receipt job |\n|---|---|---|---|---|\n'
block+='\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in rows)+'\n'+end+'\n\n'
def rewrite(text):
    a=text.index('## Projections and verify dense');b=text.index('## Glue and routing',a);section=text[a:b]
    if tag in section:
        start=section.index(tag);finish=section.find(end,start)
        if finish<0:
            finish=section.find('\n\n',section.index('|---|',start))
            finish=len(section) if finish<0 else finish
        else:finish+=len(end)
        section=section[:start]+block+section[finish:].lstrip('\n')
        return text[:a]+section.rstrip()+'\n\n'+text[b:]
    return text[:a]+section.rstrip()+'\n\n'+block+text[b:]
if rows:
    before=F.read_text();F.write_text(rewrite(before))
    head=subprocess.run(['git','show','HEAD:docs/FINDINGS.md'],cwd=R,check=True,capture_output=True,text=True).stdout
    patch=''.join(difflib.unified_diff(head.splitlines(True),rewrite(head).splitlines(True),fromfile='a/docs/FINDINGS.md',tofile='b/docs/FINDINGS.md'))
    if '--stage' in sys.argv and patch:
        with tempfile.NamedTemporaryFile(mode='w',prefix='p10-findings-',suffix='.patch') as f:
            f.write(patch);f.flush();subprocess.run(['git','apply','--cached',f.name],cwd=R,check=True)
    print('findings rows',len(rows),'staged' if '--stage' in sys.argv else 'working tree only')
