"""Receipt audit: old per-row filters vs score-aligned experimental selection."""
import json
from pathlib import Path
import decide,patches
from score_gate import legacy_summary
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    chosen=json.loads((H/'generated/selection.json').read_text())
    old=json.loads((H/'old_gate_evidence.json').read_text())
    previous,old_evidence=old['selected'],old['evidence']
    index={(e.get('arm'),tuple(e.get('row',[]))):e for e in old_evidence if 'arm' in e}
    changes=[]
    for e in chosen['evidence']:
        if 'arm' not in e:continue
        k=e['arm'],tuple(e.get('row',[]));before=index.get(k)
        if before is None:continue
        if bool(before['wins'])!=bool(e['wins']):changes.append(dict(arm=e['arm'],row=e.get('row'),old_nomination=before['wins'],new_nomination=e['wins'],R_shape_or_stage=e.get('R'),CI95_shape_or_stage=e.get('CI95'),CI95_stage_approx=e.get('CI95_approx'),six_load_CI95=None,reason='selection only; composed candidate requires held-out six-load confirmation'))
    layer_old=json.loads((H/'old_layer_gate_evidence.json').read_text())
    old_names={s['name'] for s in layer_old['selected']}
    for e in chosen['evidence']:
        if e.get('arm','').startswith('layer_'):
            changes.append(dict(e,old_nomination=e['arm'] in old_names,new_nomination=e.get('wins',False),six_load_CI95=None))
    s15=json.loads((H.parent/'s15/281b1383_report.json').read_text())
    all_nominations=[]
    for e in chosen['evidence']:
        if 'arm' not in e:continue
        key=e['arm'],tuple(e.get('row',[]));before=index.get(key)
        is_layer=e['arm'].startswith('layer_')
        def selected_in(specs):return any(sp['name']==e['arm'] and (not e.get('row') or e['row'] in sp['rows']) for sp in specs)
        all_nominations.append(dict(e,old_nomination=(e['arm'] in old_names if is_layer else before['wins'] if before else None),new_nomination=e.get('wins',False),old_selected=selected_in(layer_old['selected'] if is_layer else previous),selected=selected_in(chosen['patches'])))
    report=dict(all_nominations=all_nominations,incumbent=patches.BASE_SHA,old_selected=previous,new_selected=chosen['patches'],changes=changes,
                S15=s15['scores'],unmeasured=[e for e in chosen['evidence'] if e.get('unmeasured')],
                notes=['Legacy S15 intervals are explicitly approximate; discarded seed pairs cannot be recovered.',
                       'Shape/stage confidence intervals are not whole-layer R intervals.',
                       'No component receipt promotes S16; its held-out composition is checked by score_gate.py.'])
    (H/'gate_rejudgment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(old_selected=[s['name'] for s in previous],new_selected=[s['name'] for s in chosen['patches']],changed_nominations=len(changes),changes=changes),indent=2))
if __name__=='__main__':main()
