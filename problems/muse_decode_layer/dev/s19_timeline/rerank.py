"""Make the ranking's judgment calls explicit; recompute only our PLAN sections."""
import json,math,re
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
# p = prior odds that the next discriminating mechanism yields a useful scalable build.
# Cost is forecast hours to that verdict (development + one batched H200 lab), not GPU runtime.
PRIORS=[('P5b','P5b_ideal_roof',.80,1),('P3a','P3a_ideal_roof',.25,4),
        ('P10','P10_ideal_roof_plus_ideal_exact12',.25,4),
        ('unassigned glue/route','glue_route_ideal_roof',.25,2),
        ('P5','P5_ideal_roof',.60,1),('P3b','P3b_ideal_roof',.05,6),
        ('P7','P7_ramp_scenario',.02,6),('P11','P11_materialization_plus_idle_scenario',.05,24)]
NOTES={
'P5b':'FA3/full plus actual v16S FA3, disjoint from P5; public ports allowed R-20261026-1; completed 26d5054f passes704nativeTASK with eight preferred cells, including L v1F .757719, v16S .763592, v16F .968115. P4 whole-layer G0 remains pending. 9dc1159a supplies ~.89 call-time nominees, not a whole-layer .89. True main-loop/epilogue work remains.',
'P3a':'554c392d wait-only R1.016674 is 5.06% of the new gap, not the 41.11% family cap. New da1fb2ee matched ladder: decode-scale conditional contrast R1.007181 (~2.19% gap); MMA+activation R1.045642 (~13.66%); nonadditive ACT/sink contrasts and slow STREAM cannot be treated as exclusive hardware costs. Prior .25 reflects a remaining structural opportunity and repeated slower DSM/reuse builds.',
'P10':'7f110a37 no G0 winner. e70024b1 native scenarios small/large R1.032884/1.038411; e4030204 split-only R1.002444 (0.75% gap), whole consumer deletion R1.019571 (5.93%), producer deletion R1.086449 (25.37%). c87c2d8a 24 dispatch arms have no nominees. Broader coding/producer algorithm remains; ideal exact12 has no decode or metadata tax and is a scenario, not a demonstrated layout.',
'unassigned glue/route':'Boundary reductions, normalization and route selection remain outside attention/projection/MoE-core levers. P7 prefetch and P11 materialization do not cover that computation. Existing micro-nominees go to P4; cheapest next verdict is calibrated boundary deletion/compute attribution. External route + expert metadata grouped; separate MoE finalize remains covered by the MoE family.',
'P5':'ca022370 finish deletion R1.015900 [1.010938,1.020887] is 4.83% gap. Consumer family stays parked at its prescribed decision; native sliding/producer mechanisms remain open. 26d5054f now measures native sliding winners O v1S .780899 and L d16S/d32S/v32S .954256/.860893/.750479. This raises p to .60 and puts native sliding second; consumer-family state is untouched and P4 confirms the nominees.',
'P3b':'3330edc2 activation/gather mechanisms supplied no supported >1% next build. The unique-work family bound remains; a new mechanism must justify a build.',
'P7':'S19 update of the same 2.36us per weight/KV-boundary ramp + actual idle scenario. CUPTI has no HBM counter. 73d2d80d ALL/EDITABLE lose and raw H2 gain overlaps duplicate G control; a405c8b4 prefetches lose. This is not a physical ceiling on all continuity algorithms.',
'P11':'Only the S19 one-store/read materialization + true idle scenario is sized. Cache hits can shrink HBM savings; retention might improve issue/reuse beyond this estimate, so a whole persistent kernel has no established universal ceiling. Old S15 +1.74% synchronization estimate is not transported as a new measurement.'}

def main():
    r=json.loads((H/'result/report.json').read_text());c=r['ceilings'];rows=[]
    for lane,key,p,cost in PRIORS:
        bound=c[key];rows.append(dict(lane=lane,ceiling_scenario=key,ceiling_R=bound['R'],ceiling_gap_pct=bound['gap_share_pct'],odds_prior=p,verdict_cost_hours_estimate=cost,priority=bound['gap_share_pct']*p/cost,basis=NOTES[lane]))
    rows.sort(key=lambda v:-v['priority'])
    for i,row in enumerate(rows,1):row['rank']=i
    out=dict(job='9c16c850f7b941a5',source_sha256=r['source_sha256'],incumbent_score_estimate=r['incumbent_score_estimate'],target=r['target'],ln_gap=r['ln_gap'],formula='ceiling percent of ln gap * subjective p / forecast hours per verdict',ranked_levers=rows,ranked_work_order=['P4']+[x['lane'] for x in rows if x['lane'].startswith('P')],notes='P4 remains the live integration/scored critical path; unassigned glue is a newly exposed decision, not a new lane or a state change. All ranks preserve lane-owned row text/state. Scenario ceilings overlap and are not additive.')
    (H/'result/ranking.json').write_text(json.dumps(out,indent=2)+'\n')
    # Work text and states are read fresh immediately before the sole replacement.
    path=R/'docs/PLAN.md';s=path.read_text()
    levers='## Levers\nS19 `9c16c850` (12 S/F shapes,3fresh seeds x3rounds,72TASK). D=ln(2.6/1.875109)=.326845;\nS19 score is lab-estimated, unscored. Ceiling=%D to4.8TB/s or989TFLOP/s roof (parentheses:1.3roof).\nOrder=ceiling*p/C; p=judgment prior, C=forecast hours/verdict including a batched lab. P4 executes first.\nNonadditive. Bases, scenarios, odds and receipts: FINDINGS "S19 timeline and lever re-rank";\n[evidence](../problems/muse_decode_layer/dev/s19_timeline/README.md).\n\n| Lever | Ceiling (% gap; basis) | p / C hours | Next decision / standing |\n|---|---|---|---|\n| P5b FA3/full +v16S | 16.8 (10.4); KV roof | .80 /1 | 26d5054f ~.76 calls; P4 G0 pending |\n| P5 native sliding | 13.1 (12.7); KV roof | .60 /1 | 26d5054f .75-.78 calls; ca022370 consumer family stays parked |\n| P3a MoE M<=32 | 41.1 (21.4); unique stream | .25 /4 | da1fb2ee decode R1.00718; wait-only5.1%D, not family cap |\n| P10 projections/coding | 37.1 (20.9) ideal exact12; current25.7 (11.1) | .25 /4 | 7f110a37/c87c2d8a no winner; e4030204 split-only0.75%D; producer/coding next |\n| Unassigned boundary compute/route | 14.1 (13.0); byte roof | .25 /2 | Uncovered6.79-7.26us above1.3roof; calibrated deletion/compute oracle |\n| P3b MoE M128/256 | 27.0 (14.3); unique stream | .05 /6 | 3330edc2 no >1% next build; parked |\n| P7 stream continuity | 25.1; S19 ramp+idle scenario | .02 /6 | 73d2d80d/a405c8b4 no large causal gain; parked |\n| P11 retained operands | 5.9; S19 materialization+idle scenario | .05 /24 | Bytes alone4.5%D; issue/reuse unmeasured; parked |\n\n'
    lever_lines=levers.splitlines(True)
    slots=[i for i,l in enumerate(lever_lines) if l.startswith('| P') or l.startswith('| Unassigned')]
    lookup={l.split('|')[1].strip().split(' ',1)[0]:l for i,l in enumerate(lever_lines) if i in slots}
    for position,row in zip(slots,rows):
        key='Unassigned' if row['lane'].startswith('unassigned') else row['lane']
        lever_lines[position]=lookup[key]
    levers=''.join(lever_lines)
    s=s[:s.index('## Levers')]+levers+s[s.index('## Ranked work'):]
    lines=s.splitlines(True);positions=[i for i,line in enumerate(lines) if re.match(r'\| \d+ \| P\w+ \|',line)]
    assert len(positions)==8
    by={lines[i].split('|')[2].strip():lines[i] for i in positions}
    for i,(position,lane) in enumerate(zip(positions,out['ranked_work_order']),1):
        lines[position]=re.sub(r'^\| \d+ \|',f'| {i} |',by[lane])
    updated=''.join(lines)
    assert len(updated)<18000,('PLAN budget',len(updated))
    # No changes outside Levers or the order/rank column of existing work rows.
    def rows(text):
        return {line.split('|')[2].strip():line.split('|',2)[2] for line in text.splitlines() if re.match(r'\| \d+ \| P\w+ \|',line)}
    assert rows(path.read_text())==rows(updated)
    path.write_text(updated)
    print('PLAN chars',len(updated),'order',out['ranked_work_order'])
    print([(x['lane'],round(x['ceiling_gap_pct'],2),round(x['priority'],3)) for x in out['ranked_levers']])
if __name__=='__main__':main()
