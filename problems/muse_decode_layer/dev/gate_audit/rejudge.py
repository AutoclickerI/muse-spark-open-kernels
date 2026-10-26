"""Rejudge retained seed pairs; do not invent missing TASK checks or covariance."""
import ast, hashlib, json, math, statistics, sys
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'s16'))
from score_gate import interval,scores
from inventory import archives,payloads


def main():
    inv=json.loads((H/'inventory.json').read_text()); exact=[];shape=[]
    for job,rec in inv['receipts'].items():
        log=(R/rec['path']);receipt=json.loads(log.read_text())
        decoded=archives(receipt['result'].get('log_tail',''))
        for tag,raw in decoded:
            if tag=='PDL_AUDIT_XZ85':
                times={c['case']:c['samples'] for c in raw['cells']}
                assert len(times)==12 and all(len(v)==16 for v in times.values())
                aa,_=scores(times,'A','A2')
                for arm in times['d1S'][0]:
                    if arm in ('A','A2'):continue
                    rr,by_load=scores(times,'A',arm);score=interval(rr,aa)
                    exact.append(dict(job=job,arm=arm,**score,per_seed_R=rr,
                        loads={k:interval(v) for k,v in by_load.items()},
                        timing_pass=score['CI95'][0]>1,old_host_gates_pass=not any(c['gates'] for c in raw['cells']),
                        TASK_every_arm_seed=False,promote=False,
                        reason='Retained whole-layer seed pairs give exact six-load timing inference. Lab checked host equivalence, not public TASK on every arm/seed.',
                        archive_base_sha256=raw['base_sha'],source_sha256=raw.get('variants',{}).get(arm,raw.get('candidate_sha'))))
            cells=raw if isinstance(raw,list) else raw.get('full',[])+raw.get('sliding',[]) if isinstance(raw,dict) else []
            for c in cells:
                if not isinstance(c,dict) or 'times' not in c or 'A' not in c['times']:continue
                label=c.get('case',str(c.get('B'))+'x'+str(c.get('L')))
                values=c['times'];n=len(values['A'])
                if n<2:continue
                aa=[a/b for a,b in zip(values.get('A2',values['A']),values['A'])]
                for arm,v in values.items():
                    if arm in ('A','A2'):continue
                    assert len(v)==n
                    rr=[a/b for a,b in zip(values['A'],v)];score=interval(rr,aa)
                    row=dict(job=job,tag=tag,case=label,arm=arm,**score,point_winner=score['R']>1,
                        scope='Measured shape/stage only; this is not six-load G0 R.',promote=False)
                    key=(job,label,arm)
                    if not any((x['job'],x['case'],x['arm'])==key for x in shape):shape.append(row)
    # Pin recovered sources for nominated whole-layer policies. These inputs
    # are original immutable submissions; none are reinterpreted as TASK passes.
    pdl=next(x for j,x in inv['receipts'].items() if j.startswith('0f74021a'))
    chunks=payloads((R/pdl['path']).with_suffix('.py').read_text())
    body=next(s for _,s in chunks if 'PDL_AUDIT_XZ85' in s)
    constants={t.id:ast.literal_eval(n.value) for n in ast.parse(body).body if isinstance(n,ast.Assign)
               for t in n.targets if isinstance(t,ast.Name) and t.id in ('PA_BASE','PA_HINT','PA_META','PA_D1')}
    nominees=[]
    for arm,key,name in [('B','PA_HINT','First-weight L2 hints plus immutable metadata before wait'),
                          ('M','PA_META','Immutable metadata before wait'),('D','PA_D1','Prior decode-1 full glue')]:
        score=next(x for x in exact if x['job'].startswith('0f74021a') and x['arm']==arm)
        source=constants[key];assert hashlib.sha256(source.encode()).hexdigest()==score['source_sha256']
        (H/'generated'/('nominee_'+arm+'.py')).write_text(source)
        nominees.append(dict(name=name,arm=arm,job=score['job'],R=score['R'],CI95=score['CI95'],
            base_sha256=score['archive_base_sha256'],source_sha256=score['source_sha256'],
            old_verdict='Closed recipe / v32 loss or inside per-load A:A',new_verdict='Six-load timing pass; nominate for current-base TASK and disjoint confirmation',
            next='Rebase exact delta to current incumbent, choose maximum R among source-wide policies, run public TASK on all arms/seeds, then confirm composed G0 on disjoint seeds.'))
    notes={
        36:('reopen_correctness','Nondeterministic rejection, 0.86% host hidden error and changed IDs do not prove a TASK failure. No immutable per-seed receipt: TASK and timing unmeasured.'),
        42:('reopen_correctness','7426e17c stopped after one three-seed d1S row at host-only 0.3% error / exact-ID gates. Whole-layer six-load R and CI unavailable. Reopen immutable-weight prefetch, do not call seven unmeasured cells losses.'),
        50:('shape_nominees','LIW/TYPW at M128; LIW/TYPW/PAIR2/AD64D/AD64P/AD64PW at M256 have sub-1 points. Means/ranges only, CI unavailable. PERMW already integrated; compare alternatives against current PERMW, not S7/LI4.'),
        54:('shape_nominees','79061182 DSM C128 d1S and DSM v1S are eligible by point; per-shape raw intervals retained below. No all-shape veto. Later S16/P5 DSM supersedes these choices. c0fd644c FP8-MMA still loses by point.'),
        56:('shape_nominees','39949536 q1 full page-flight: B16 new/old .99744, B32 .99664. Three-seed summaries/ranges only, exact CI unavailable; nominate on current full-attention base. FP16 PV/raw-V/LDSM losses unchanged.'),
        57:('dominated_nominee','cf79aa07 B16 pair2/3 new/S13 .99754/.99639 are eligible shape points, but measured best128 is .92578; do not replace the faster current N128. Stage3 loses; stage4 fold already S15.'),
        58:('dominated_nominee','c773699a d32 fused F128 new/A .9684 vs split C128 .9679: split remains faster. Compare shape alternatives to best measured dispatch. Full-check/twelve-shape data absent; component LAST/MASK wins already followed by P5.'),
        59:('measured_losses_incomplete','TMA verify GEMM/stats/O-only measured verify rows lose. Host-equivalence rejections are unmeasured, not TASK failures. No per-seed twelve-shape data to manufacture R/CI.'),
        60:('measured_losses_incomplete','Producer-glue full variants lose on every measured shape; metadata-only d1 1.177 also loses. Traces are repeated profile spans, not independent paired seed times. Do not use them as CI samples.'),
        61:('timing_reversal','0f74021a B/M/D pass six-load timing CI; F and 8db42132 all-register recipe still lose. TASK on lab arms remains unmeasured.'),
        62:('shape_nominees','cpall .996850 was a control-normalized d1 number; raw off/A is 1.010536. Drop the A:A mean correction. Direct qkv3 removal .981311 at d1 and bal2 removal .982068 at d16 remain eligible per-load selections (already reflected in later dispatch); summarized data cannot recover seed covariance/CI.')}
    rows=[]
    for i,row in enumerate(inv['rows']):
        status,why=notes.get(i,('unchanged_scope','No old all-load/host-equivalence reversal established by retained receipts; keep only the measured recipe conclusion. Component losses are not a family closure.'))
        rows.append(dict(**row,audit_status=status,new_verdict=why))
    result=dict(method='Exact score_gate.interval on log paired seed R, max paired/A:A SE. Shape statistics never stand in for G0. No A:A mean subtraction; no missing covariance or TASK passes invented.',
        closed_rows=len(rows),primary_receipts=len(inv['receipts']),whole_layer=exact,shape_pairs=shape,nominees=nominees,rows=rows,
        missing='Most older receipts keep means/ranges or profile repetitions. R and CI are null/absent where six-load seed pairs are unavailable. Official host pass covers only that host, not every lab arm.')
    (H/'rejudgment.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Closed-recipe score-gate audit','',f"Audited {len(rows)} FINDINGS rows and {len(inv['receipts'])} primary receipts dated 2026-10-23 onward. Source predicates and available raw arrays are indexed in `inventory.json`; all row verdicts and exact statistics are in `rejudgment.json`.",'',
        'Per seed, R is the six-load geometric mean of old/new, with 46 sliding + 16 full per load. Intervals use the shared score gate and A:A noise. Archived component/profile repetitions are not substituted for paired input seeds. No host-equivalence or exact-ID criterion remains a promotion condition.','',
        '| Recipe / receipt | Old decision | New timing R [95% CI] | New decision |','|---|---|---|---|']
    for x in exact:
        label={'B':'hints + metadata','M':'metadata','D':'d1-full glue','F':'register preloads'}.get(x['arm'],x['arm'])
        lines.append(f"| {label}, `{x['job'][:8]}`/{x['arm']} | closed / per-load veto | {x['R']:.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}] | {'timing pass; nominate' if x['timing_pass'] else 'timing loss'} |")
    lines+=['','B/M/D used 16 held-out inputs `2100911 + 104729*i` on S14 synthetic weights seed 5353 / router 197731. They passed the old lab host-agreement checks. Those checks are not public TASK; no full promotion is asserted. D is the prior d1-full glue policy, not a combination of B and M. Rank source-wide policies by R: M, D, B, F.','',
        'Shape nominees: q1 full page-flight (B16/B32); heavy LIW/TYPW and M256 PAIR2/adaptive candidates against current PERMW. Their receipts retain means/ranges rather than six-load seed pairs, so exact G0 CI is unavailable. DSM and register-fold winners are already represented in S16/S17. B16 pair-register V loses to the measured N128 comparator even though it beats the older S13 point.','',
        'Reopened correctness: in-kernel own-weight prefetch `7426e17c` and the old persistent-qkv recipe. Their host-error / exact-ID rejection does not establish a TASK failure. Missing timing stays unmeasured. The higher-impact decode-once/producer-consumer, queue, persistent and TMA recipes retain their large measured losses; relaxing a gate does not make those samples faster.','',
        'Reproduce: `python3 problems/muse_decode_layer/dev/gate_audit/inventory.py` then `python3 problems/muse_decode_layer/dev/gate_audit/rejudge.py`. The compressed archives come from immutable `runs/` JSON; generated extracted sources/logs stay ignored.']
    (H/'README.md').write_text('\n'.join(lines)+'\n')
    print('Exact whole-layer',[(x['job'][:8],x['arm'],round(x['R'],6),x['CI95'],x['timing_pass']) for x in exact]);print('Shape pairs',len(shape),'nominees',len(nominees))

if __name__=='__main__':main()
