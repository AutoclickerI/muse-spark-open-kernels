"""Read exact reset receipts and emit S16 plus its auditable selection manifest.

Stage wins nominate a second whole-layer experiment; they never freeze S16.
The existing S15 report is the mandatory admission gate for the integration base.
"""
import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import lzma
import math
import re
import statistics
import struct
import subprocess
import sys
from pathlib import Path
import patches

H=Path(__file__).resolve().parent;R=H.parents[3]


def archive(log,tag):
    rows=[line.split(tag+' ',1)[1].strip() for line in log.splitlines() if line.startswith(tag+' ')]
    assert len(rows)==1,('missing/truncated archive',tag)
    return json.loads(lzma.decompress(base64.b85decode(rows[0])))


def paired(values,control,min_n=3):
    assert len(values)>=min_n and len(values)==len(control)
    assert all(math.isfinite(v) and v>0 for v in values+control)
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values))
    noise=max(abs(x-1) for x in control)
    return dict(ratio=mean,se=se,n=len(values),AA_radius=noise,wins=mean+1.96*se<1-noise)


def read_receipt(entry,files):
    matches=[]
    for path in files:
        source=path.with_suffix('.py')
        if not source.exists():continue
        if hashlib.sha256(source.read_bytes()).hexdigest()==entry['sha256']:matches.append(path)
    assert len(matches)==1,(entry['file'],'receipt absent or ambiguous',len(matches))
    path=matches[0];r=json.loads(path.read_text())
    assert r.get('result',{}).get('passed'),(entry['file'],'official failed')
    log=r['result'].get('log_tail','')
    assert 'LAB_ERROR' not in log and 'LAB_TRUNCATED' not in log,(entry['file'],'incomplete lab')
    return dict(path=str(path.relative_to(R)) if path.is_relative_to(R) else str(path),job=r['id'],log=log,sha256=entry['sha256'])


def decode_boundary(log):
    raw=lzma.decompress(base64.b85decode(next(line.split(' ',1)[1] for line in log.splitlines() if line.startswith('BD_BATCH_XZ85 '))))
    n=struct.unpack_from('<I',raw)[0];data=json.loads(raw[4:4+n]);binary=struct.unpack('<'+'I'*data['binary_count'],raw[4+n:])
    for cell in data['confirmation']['cells']+data['boundaries']:
        names=cell['sample_columns'];offset=cell['sample_offset'];samples=[]
        for i in range(cell['sample_rows']):
            row=dict(zip(names,binary[offset+i*len(names):offset+(i+1)*len(names)]));a=row['A']
            samples.append({k:(v if k=='A' else a+((v>>1)^-(v&1)))/1000 for k,v in row.items()})
        cell['samples']=samples
    return data


def nominate(receipts):
    registry=patches.registry();choices={};evidence=[]
    def candidate(name,row,proof):
        evidence.append(dict(arm=name,row=list(row),**proof))
        if not proof['wins']:return
        spec=copy.deepcopy(registry[name]);spec['rows']=[list(row)]
        family=spec['family']
        # Attention families compete at a shape; epi/heavy policies compete at a token count.
        key=('attention',tuple(row)) if family in ('global','dsm','early','soft') else ('policy',name) if family in ('pdl','boundary') else (family,row[0]*row[1])
        if key in choices and choices[key][0]['name']==name:
            if list(row) not in choices[key][0]['rows']:choices[key][0]['rows'].append(list(row))
        elif key not in choices or proof['ratio']<choices[key][1]:choices[key]=(spec,proof['ratio'])
    def parse_global():
        global_rows=archive(receipts['global']['log'],'CS_XZ85')
        assert len(global_rows)==4
        for cell in global_rows:
            assert cell['seeds']==16 and not any(not v[0] for v in cell['gate'].values()) if cell.get('gate') else cell['seeds']==16
            label=cell['case'];row=(int(re.search(r'\d+',label)[0]),1,label.endswith('S'))
            reference='OLD'  # last-CTA sliding / direct-partials full = S15 attention
            times=cell['times'];aa=[b/a for a,b in zip(times['A'],times['A2'])]
            for kind in ('C64','C128','W4','W8'):
                rr=[x/y for x,y in zip(times[kind],times[reference])]
                candidate('global_'+kind,row,paired(rr,aa,16))

    def parse_early():
        early=archive(receipts['early']['log'],'EH_XZ85')
        assert early['seeds']==16
        aa=[x/y for x,y in zip(early['times']['A2'],early['times']['A'])]
        for kind in ('E4','E2','E1','L4'):
            candidate('early_'+kind,(1,1,True),paired(early['ratios']['OLD4'][kind],aa,16))

    def parse_dsm():
        dsm=archive(receipts['dsm']['log'],'DS_XZ85')
        assert len(dsm)==3
        for cell in dsm:
            assert cell['seeds']==8 and len(cell['gates'])==8
            for name,rr in cell['ratios'].items():
                if name.startswith('C'):candidate('dsm_'+name[1:],(cell['B'],8,True),paired(rr,cell['ratios']['A2'],8))

    def parse_soft():
        # Softmax reset probes have three seeds. The full S16 test compares their N64
        # stage-6 path with S15's N128 stage-4 path; component wins are nominations only.
        for line in receipts['soft']['log'].splitlines():
            match=re.match(r'R B(16|32) L8 ',line)
            if not match:continue
            numbers={name:(float(mean),float(lo),float(hi)) for name,mean,lo,hi in re.findall(r'(\w+)=[\d.]+/r([\d.]+)\[([\d.]+),([\d.]+)\]',line)}
            assert 'A2' in numbers
            noise=max(abs(numbers['A2'][1]-1),abs(numbers['A2'][2]-1))
            for kind in ('mix25','mix50','gap','mixgap','tc25','ping','ping25'):
                if kind in numbers:
                    mean,lo,hi=numbers[kind]
                    candidate('soft_'+kind,(int(match[1]),8,False),dict(ratio=mean,range=[lo,hi],AA_radius=noise,n=3,wins=hi<1-noise))

    def parse_epi():
        epi=receipts['epi']['log']
        assert re.search(r'END seeds=16\b',epi) and not re.search(r'^FAIL ',epi,re.M)
        for m in (1,8,16,32):
            noise_match=re.search(rf'^AA M{m} abs=([\d.]+) us',epi,re.M);assert noise_match
            aa_us=float(noise_match[1])
            for kind in ('VEC','PROD','NORM','PAIR'):
                match=re.search(rf'^PAIR M{m} {kind} us=([\d.]+) r=([\d.]+) se=([\d.]+)',epi,re.M);assert match
                us,mean,se=map(float,match.groups());radius=aa_us/max(us/mean,1e-12)
                proof=dict(ratio=mean,se=se,n=16,AA_radius=radius,wins=mean+1.96*se<1-radius)
                b,q=(1,8) if m==8 else (m,1)
                for w in (True,False):candidate('epi_'+kind,(b,q,w),proof)

    def parse_heavy():
        heavy=receipts['heavy']['log'];assert re.search(r'END seeds=16\b',heavy)
        aa_match=re.search(r'AA signed/abs/max=[\d.+-]+/[\d.]+/([\d.]+)',heavy);assert aa_match
        aa_us=float(aa_match[1])
        for m in (128,256):
            baseline=re.search(rf'^M{m} S14_NF n=48 us=([\d.]+)',heavy,re.M);assert baseline
            radius=aa_us/float(baseline[1])
            for kind in ('DIRECT_CLUSTER1','DIRECT_NORMAL','DIRECT_SERIAL','DEEP_SERIAL'):
                match=re.search(rf'^M{m} {kind} n=48 us=[\d.]+ pair=([\d.]+) lo/hi=([\d.]+)/([\d.]+).*TASK=PASS',heavy,re.M);assert match
                mean,lo,hi=map(float,match.groups())
                for w in (True,False):candidate('heavy_'+kind,(m//8,8,w),dict(ratio=mean,range=[lo,hi],AA_radius=radius,n=48,wins=mean<1-radius and hi<1))

    def parse_boundary():
        boundary=decode_boundary(receipts['boundary']['log'])
        for family,cells,arms in [('pdl',boundary['confirmation']['cells'],('B','M','D')),('boundary',boundary['boundaries'],('R','P','RP','C','S'))]:
            for kind in arms:
                spec=registry[family+'_'+kind];rows=[];proofs=[]
                for cell in cells:
                    assert not cell['gates'] and len(cell['samples'])==16
                    label=cell['case'];row=(int(re.search(r'\d+',label)[0]),1 if label.startswith('d') else 8,label.endswith('S'))
                    if list(row) not in [list(x) for x in spec['rows']]:continue
                    samples=cell['samples'];aa=[s['A2']/s['A'] for s in samples]
                    proof=paired([s[kind]/s['A'] for s in samples],aa,16)
                    rows.append(row);proofs.append(proof)
                # These source-wide transformations have not been selectively
                # specialized; reject a policy with any affected regression/noise.
                if rows and all(p['wins'] for p in proofs):
                    for row,proof in zip(rows,proofs):candidate(family+'_'+kind,row,proof)
                else:evidence.append(dict(arm=family+'_'+kind,wins=False,reason='not every measured affected row clears A:A',rows=[list(r) for r in rows]))

    for name,parse in ((name,fn) for name,fn in [('global',parse_global),('early',parse_early),('dsm',parse_dsm),('soft',parse_soft),('epi',parse_epi),('heavy',parse_heavy),('boundary',parse_boundary)]):
        if name not in receipts:
            evidence.append(dict(batch=name,wins=False,reason='receipt unavailable or official/lab failure; no candidate admitted'))
            continue
        previous=copy.deepcopy(choices)
        try:parse()
        except (AssertionError,KeyError,ValueError) as exc:
            # Do not lose independent valid batches because this host returned
            # incomplete gates or a different archive schema. Preserve the cause.
            choices.clear();choices.update(previous)
            evidence.append(dict(batch=name,wins=False,reason=type(exc).__name__+':'+str(exc),unmeasured=True))
    grouped={}
    for spec,ratio in choices.values():
        if spec['name'] not in grouped:grouped[spec['name']]=spec
        elif spec['rows'][0] not in grouped[spec['name']]['rows']:grouped[spec['name']]['rows']+=spec['rows']
    # M/B are mutually exclusive global preload transformations; R/RP and P/RP
    # are alternative policies, not independent effects. Choose the measured best.
    policies=[v for v in grouped.values() if v['family'] in ('pdl','boundary')]
    if policies:
        keep=min(policies,key=lambda s:statistics.mean(e['ratio'] for e in evidence if e['arm']==s['name'] and e.get('wins') and 'ratio' in e))
        for s in policies:
            if s is not keep:grouped.pop(s['name'],None)
    return list(grouped.values()),evidence


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipts',type=Path,nargs='*');ap.add_argument('--out',type=Path,default=H/'generated/selection.json');a=ap.parse_args()
    files=a.receipts or list((R/'runs').glob('*/*.json'))
    inventory=json.loads((H/'reset_inventory.json').read_text());receipts={}
    keys={'0ccc2548':'global','f1c270e6':'dsm','03086413':'soft','917e79f3':'epi','f867a8b7':'heavy','d29911c5':'boundary','49ce378b':'s15','0cec6706':'early'}
    unavailable={}
    for entry in inventory:
        name=keys[entry['sha256'][:8]]
        try:receipts[name]=read_receipt(entry,files)
        except AssertionError as exc:
            if name=='s15':raise
            unavailable[name]=str(exc)
    # Run exactly the existing S15 freeze gate. No source is emitted on failure.
    subprocess.run([sys.executable,str(H.parent/'s15/report.py'),receipts['s15']['job']],cwd=R,check=True,capture_output=True)
    report=json.loads((H.parent/'s15'/(receipts['s15']['job'][:8]+'_report.json')).read_text())
    assert report['freeze_gate'],'S15 promotion gate failed; S16 base is not admitted'
    frozen=R/'problems/muse_decode_layer/cand_r3_best7_s15_final.py'
    source=patches.BASE.read_text();assert hashlib.sha256(source.encode()).hexdigest()==report['source_sha256']
    frozen.write_text(source)
    selected,evidence=nominate(receipts)
    assert selected,'no executable reset arm cleared its measured A:A envelope'
    admission=json.loads((H/'admission.json').read_text())
    for spec in selected:
        assert all(admission[spec['name']][m]['passed'] for m in ('normal','forced')),spec['name']
    source,meta=patches.compose(selected)
    out=H/'generated/s16.py';out.write_text(source)
    result=dict(meta,source=str(out.relative_to(R)),status='experimental; combined local screen and whole-layer A/B required',
                reset_receipts={k:{name:v for name,v in row.items() if name!='log'} for k,row in receipts.items()},unavailable_receipts=unavailable,evidence=evidence)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(out,meta['sha256'])


if __name__=='__main__':main()
