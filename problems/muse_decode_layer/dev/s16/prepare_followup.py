"""Select the next pair from frozen S16 off arms and newly gated attention."""
import hashlib,json,subprocess,sys
from pathlib import Path
import patches
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    report=json.loads((H/'s16_report.json').read_text());assert report['freeze_gate'] and report['job']=='37c2879f62554dff'
    archived=json.loads((H/'batches/504e1fa66298bf9e/selection.json').read_text())
    source,meta=patches.compose(archived['patches']);assert meta['sha256']==report['source_sha256']
    baseline=H/'generated/frozen_s16.py';baseline.write_text(source)
    selected=H/'generated/attention_selection.json'
    subprocess.run([sys.executable,str(H/'decide.py'),'--out',str(selected)],cwd=R,check=True)
    data=json.loads(selected.read_text());removed={name for name,r in report['pieces'].items() if r['R']<1}
    data['patches']=[s for s in data['patches'] if s['name'] not in removed]
    assert {s['name'] for s in data['patches']}=={'global_OLD','global_PP','epi_PAIR','layer_release_attention','layer_coop_v1'}
    data.update(selection_seed_ids=report['raw']['seed_ids'],baseline_arm='S16',baseline_sha256=hashlib.sha256(source.encode()).hexdigest(),
                selection_job=report['job'],removed_by_off_arm=sorted(removed),
                decision='Point-estimate pruning selects a new composition; only a fresh held-out six-load pair can promote it over S16.')
    out=H/'generated/followup_selection.json';out.write_text(json.dumps(data,indent=2)+'\n')
    (H/'followup_decision.json').write_text(json.dumps({k:v for k,v in data.items() if k in ('patches','selection_seed_ids','baseline_arm','baseline_sha256','selection_job','removed_by_off_arm','decision')},indent=2)+'\n')
    print('READY',out,'baseline',baseline,flush=True)


if __name__=='__main__':main()
