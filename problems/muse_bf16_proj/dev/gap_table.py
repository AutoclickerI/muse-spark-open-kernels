"""Receipt-derived projection gap, normalized by each run's paired SGLang ratio."""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BOARD='f3786341611f4f1f'
LAYERS=('qkv_gate','o','pooler','unpooler','router')
MS=(1,16,32,8,128,256)


def analyze():
    jobs=[]
    for path in sorted((ROOT/'runs/muse_bf16_proj').glob('*.json')):
        job=json.loads(path.read_text());result=job.get('result') or {}
        if not result.get('passed') or result.get('flags') or result.get('gpu')!='H200':continue
        rows={(r['case']['layer'],r['case']['m']):r for r in result.get('records',[]) if r.get('event')=='benchmark'}
        if rows:jobs.append((job['id'],rows,path))
    canonical=next(rows for name,rows,_ in jobs if name==BOARD)
    output=[];loads=[]
    for m in MS:
        for layer in LAYERS:
            key=(layer,m);base=canonical[key]['baseline_ns']/1000
            job,record=max(((name,rows[key]) for name,rows,_ in jobs if key in rows),key=lambda v:v[1]['speedup'])
            best=base/record['speedup'];target=base/1.26
            output.append(dict(layer=layer,m=m,sglang_us=base,board_us=canonical[key]['median_ns']/1000,
                               best_normalized_us=best,best_receipt_us=record['median_ns']/1000,
                               best_speedup=record['speedup'],best_job=job,uniform_126_excess_us=max(0,best-target)))
        group=output[-5:];sg=sum(r['sglang_us'] for r in group);board=sum(r['board_us'] for r in group);best=sum(r['best_normalized_us'] for r in group)
        loads.append(dict(m=m,sglang_us=sg,board_us=board,best_us=best,best_speedup=sg/best,
                          budget_126_us=sg/1.26,save_to_uniform_126_us=max(0,best-sg/1.26)))
    best_score=math.prod(r['best_speedup'] for r in loads)**(1/6)
    board_score=math.prod(r['sglang_us']/r['board_us'] for r in loads)**(1/6)
    only_verify_factor=(1.26/best_score)**3
    report={'receipts':len(jobs),'board_score':board_score,'optimistic_stitched_score':best_score,
            'verify_only_speed_factor_to_126':only_verify_factor,
            'verify_only_speed_factor_from_board':(1.26/board_score)**3,'shapes':output,'loads':loads}
    return report


def markdown(report):
    lines=['<!-- PROJ_GAP_BEGIN -->',
           'H200 gap to 1.26x, from all '+str(report['receipts'])+' clean completed receipts. SGLang and board',
           'times use scored `'+BOARD+'`. Best = that SGLang time / the highest observed',
           'paired speedup for this shape; it is a common-baseline equivalent, not a new measured',
           'median. Choosing the maximum across receipts is optimistic and includes timing noise;',
           'the new portfolio must reproduce any gain. All times below are microseconds.',
           '', '| m | Layer | SGLang | Board | Best equivalent | Best receipt median | Receipt | Excess over SGLang/1.26 |',
           '|---:|---|---:|---:|---:|---:|---|---:|']
    for r in report['shapes']:
        lines.append(f"| {r['m']} | {r['layer']} | {r['sglang_us']:.3f} | {r['board_us']:.3f} | {r['best_normalized_us']:.3f} | {r['best_receipt_us']:.3f} | `{r['best_job']}` | {r['uniform_126_excess_us']:.3f} |")
    lines += ['', 'The last column is a sufficient uniform-shape target, not the score formula. The',
              'official projection score is the geometric mean of six *load* ratios; each load sums',
              'its five equally weighted projections (weight 62 cancels). The overall-board formula',
              'remains unpublished; the earlier 11% projection weight is a model, not a rule.',
              '', '| m/load | SGLang sum | Board sum | Best equivalent sum | Best load ratio | Budget at 1.26x | Saving to that budget |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['loads']:
        lines.append(f"| {r['m']} | {r['sglang_us']:.3f} | {r['board_us']:.3f} | {r['best_us']:.3f} | {r['best_speedup']:.3f}x | {r['budget_126_us']:.3f} | {r['save_to_uniform_126_us']:.3f} |")
    lines += ['',f"The optimistic stitched score is **{report['optimistic_stitched_score']:.6f}x**, versus the board's",
              f"{report['board_score']:.6f}x. If the other four loads hold, m128 and m256 each need",
              f"another **{report['verify_only_speed_factor_to_126']:.3f}x** speedup to reach 1.26x overall.",
              f"Starting from the actual board rather than optimistic stitching, that factor is **{report['verify_only_speed_factor_from_board']:.3f}x**.",
              'The largest uniform-budget deficits are pooler/unpooler at m16/32/128/256, then qkv and',
              'other BF16 layers. Router fusion alone cannot close the complete gap; tune all 30',
              'shapes, especially memory-heavy BF16 projections. Receipt/source pairs in `runs/`',
              'own these measurements; profiles diagnose kernels but do not replace paired medians.',
              '<!-- PROJ_GAP_END -->']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--write',action='store_true');args=ap.parse_args()
    report=analyze();body=markdown(report)
    (Path(__file__).with_name('gap_table_local.json')).write_text(json.dumps(report,indent=2)+'\n')
    if args.write:
        path=ROOT/'docs/FINDINGS.md';text=path.read_text()
        if '<!-- PROJ_GAP_BEGIN -->' in text:
            a=text.index('<!-- PROJ_GAP_BEGIN -->');b=text.index('<!-- PROJ_GAP_END -->',a)+len('<!-- PROJ_GAP_END -->\n')
        else:
            a=text.index('| Layer | m | SGLang us | Scored us | Speedup | Floor us |');b=text.index('\n| Finding |',a)
        path.write_text(text[:a]+body+'\n'+text[b:])
    print(body)
