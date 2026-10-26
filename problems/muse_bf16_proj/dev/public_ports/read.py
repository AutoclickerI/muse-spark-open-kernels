"""Retain every port sample; independent seed means and A:A Student-t CIs."""
import ast,base64,csv,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
from promotion import interval
exec((H.parent/'native_boundary/codec.py.inc').read_text())

def ci(values,aa=None):
    v=interval(values,aa);return [v['R'],*v['CI95']]

def analyse(receipt):
    packets=read_packets(receipt['result']['log_tail'])
    assert packets and all(not p['packet'].get('transport_errors') and p['packet'].get('complete') is not False for p in packets),'failed transport'
    blob=next(s.split(' ',1)[1] for s in result_lines(receipt['result']['log_tail']) if s.startswith('PUBLIC_PROJ_XZ85 '))
    raw=nb_time_decode(blob);assert raw['complete'],raw.get('fatal')
    manifest=json.loads((H/'lab_manifest.json').read_text())
    assert list(raw['seeds'])==manifest['seeds'] and raw['rounds']==manifest['rounds']
    assert any(p['source_sha256']==manifest['payload_sha256'][:16] for p in packets)
    cells=[];nominees=[]
    for row in raw['rows']:
        times={name:[statistics.mean(z[name])*.008 for z in row['times']] for name,status in row['status'].items() if status=='ready'}
        assert len(row['times'])==8 and all(len(z[name])==4 for z in row['times'] for name in times)
        a,aa=times['A'],times['AA'];control=[y/x for x,y in zip(a,aa)];arms={}
        for name,status in row['status'].items():
            if status!='ready':arms[name]=dict(measured=False,reason=status);continue
            v=times[name];arms[name]=dict(measured=True,ratio=ci([y/x for x,y in zip(a,v)],control),us=statistics.mean(v),saved_us=statistics.mean(x-y for x,y in zip(a,v)))
        eligible=[n for n in ('O','P','C','N') if arms[n]['measured'] and not(n=='N' and row['same_S19_NF'])]
        best=min(eligible,key=lambda n:arms[n]['ratio'][0]) if eligible else None
        cell=dict(t=row['t'],role=row['role'],kind=row['kind'],inactive=row['inactive'],same_S19_NF=row['same_S19_NF'],baseline_us=statistics.mean(a),arms=arms,best=best)
        cells.append(cell)
        if best and arms[best]['ratio'][0]<1 and not row['inactive']:nominees.append(dict(t=row['t'],role=row['role'],provider=best,**arms[best]))
    return dict(job=receipt['id'],raw=raw,cells=cells,nominees=nominees,limits='Selection-stage time ratio candidate/S19, lower is faster. Arithmetic mean of all4 symmetric rounds per fresh seed. Student-t CI uses max(primary,A:A) log SE. Actual paid S19 norm/router/gate neighbours included; inactive cells diagnostic. Nominees require P4 whole-layer held-out six-load G0, no per-site veto.')

if __name__=='__main__':
    p=Path(sys.argv[1])
    if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
    wrapper=p.with_suffix('.py').read_text();tree=ast.parse(wrapper);payloads=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(a,'id','')=='_LAB_PAYLOADS' for a in n.targets)));assert any(hashlib.sha256(x[2].encode()).hexdigest()==json.loads((H/'lab_manifest.json').read_text())['payload_sha256'] for x in payloads)
    out=analyse(json.loads(p.read_text()));(H/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    with (H/'ratios.csv').open('w') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(['T','site','arm','ratio','CI_low','CI_high','S19_us','candidate_us','inactive','measured'])
        for c in out['cells']:
            for n,a in c['arms'].items():w.writerow([c['t'],c['role'],n,*(a.get('ratio') or ['','','']),c['baseline_us'],a.get('us'),c['inactive'],a['measured']])
    lines=['# Public projection port ranking','',out['limits'],'',f"Receipt `{out['job']}`; {out['raw']['checks']} projection, {out['raw']['router_checks']} router, {out['raw']['gate_layer_checks']} layer TASK checks. Baseline prepare {out['raw']['base_prepare_s']:.3f}s; added banks {out['raw']['added_prepare_s']:.3f}s.",'','| T | Site | Raw O | PDL P | Coded C | NF N |','|---:|---|---|---|---|---|']
    for c in out['cells']:
        def fmt(n):
            a=c['arms'][n]
            if not a['measured']:return 'unmeasured'
            v,l,h=a['ratio'];return f'{v:.4f} [{l:.4f},{h:.4f}]'+(' same S19' if n=='N' and c['same_S19_NF'] else '')
        lines.append('| '+str(c['t'])+' | '+c['role']+(' (inactive)' if c['inactive'] else '')+' | '+' | '.join(map(fmt,('O','P','C','N')))+' |')
    (H/'PORTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(job=out['job'],nominees=out['nominees'],prepare=out['raw']['prepare_times'],failures=out['raw']['failures']),indent=2))
