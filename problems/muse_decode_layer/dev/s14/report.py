"""Exact-receipt S14 leave-one-out selection, with each piece's unchanged loads checked too."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path
H=Path(__file__).resolve().parent
R=H.parents[3]
LOADS=('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('--manifest',type=Path,default=H/'s14ab_manifest.json');a=ap.parse_args()
    paths=list((R/'runs/muse_decode_layer').glob('*_'+a.job+'*.json'));assert len(paths)==1,paths
    receipt=json.loads(paths[0].read_text());manifest=json.loads(a.manifest.read_text())
    assert hashlib.sha256(paths[0].with_suffix('.py').read_bytes()).hexdigest()==manifest['sha256']
    log=receipt['result']['log_tail'];loads={load:{} for load in LOADS};pairs={load:{} for load in LOADS}
    for load,arm,ratio in re.findall(r'^LOAD ((?:decode|verify) \d+) (\w+) r=([\d.]+)$',log,re.M):loads[load][arm]=float(ratio)
    aa={load:dict(mean=float(mean),lo=float(lo),hi=float(hi),se=float(se),n=int(n)) for load,mean,lo,hi,se,n in re.findall(r'^AA ((?:decode|verify) \d+) mean=([\d.]+) lo=([\d.]+) hi=([\d.]+) se=([\d.]+) n=(\d+)$',log,re.M)}
    for load,arm,mean,se,norm,normse,n in re.findall(r'^PAIR ((?:decode|verify) \d+) (\w+) mean=([\d.]+) se=([\d.]+) norm=([\d.]+) normse=([\d.]+) n=(\d+)$',log,re.M):
        pairs[load][arm]=dict(mean=float(mean),se=float(se),norm=float(norm),normse=float(normse),n=int(n))
    failures=[line for line in log.splitlines() if any(x in line for x in (' FAIL','INCOMPLETE','LAB_ERROR','ERROR ','LAB_TRUNCATED'))]
    shapes=re.findall(r'^L (\w+) (\w+) r=[\d.]+ se=[\d.]+ d=[+\-\d.]+ A=[\d.]+ n=(\d+)',log,re.M)
    complete=bool(receipt['result'].get('passed') and 'END shapes=12/12' in log and not failures and len(aa)==6 and all(r['n']==16 for r in aa.values()) and shapes and all(int(n)==16 for _,_,n in shapes))
    decisions={};controls={}
    for arm,spec in manifest['arms'].items():
        decisions[arm]={};controls[arm]={}
        for load in LOADS:
            t=int(load.split()[1])*(8 if load.startswith('verify') else 1);noise=aa.get(load);radius=max(abs(noise['lo']-1),abs(noise['hi']-1)) if noise else math.inf
            p=pairs[load].get(arm);ratio=p['mean'] if p else loads[load].get(arm)
            if t in spec['active_tokens']:
                decisions[arm][load]=dict(ratio=ratio,AA_radius=radius,wins=bool(ratio is not None and ratio>1+radius),paired=p)
            else:controls[arm][load]=dict(ratio=ratio,AA_radius=radius,within_AA=bool(ratio is not None and abs(ratio-1)<=radius))
    retained=[p['name'] for p in manifest['patches']]
    pieces_win=all(arm in decisions and all(r['wins'] for r in decisions[arm].values()) for arm in retained)
    controls_pass=all(r['within_AA'] for rows in controls.values() for r in rows.values())
    changed_win=all(r['wins'] for r in decisions.get('S13',{}).values())
    report=dict(job=receipt['id'],source_sha256=manifest['host_sha256'],complete=complete,loads=loads,A_A=aa,decisions=decisions,controls=controls,failures=failures,pieces_win=pieces_win,controls_pass=controls_pass,changed_loads_win=changed_win,freeze_gate=bool(complete and pieces_win and controls_pass and changed_win))
    if all('S13' in row for row in loads.values()):report['six_load_S13_over_S14']=math.exp(sum(math.log(row['S13']) for row in loads.values())/6)
    (H/(receipt['id'][:8]+'_report.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
