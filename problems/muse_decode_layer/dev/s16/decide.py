"""Read exact reset receipts and emit S16 plus its auditable selection manifest.

Measured shape ratios and six-load policy R nominate an experiment; only its
held-out confirmation can promote it. Exact S15 60254883 is the frozen base.
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
from score_gate import interval, scores

H=Path(__file__).resolve().parent;R=H.parents[3]


def archive(log,tag):
    rows=[line.split(tag+' ',1)[1].strip() for line in log.splitlines() if line.startswith(tag+' ')]
    assert len(rows)==1,('missing/truncated archive',tag)
    return json.loads(lzma.decompress(base64.b85decode(rows[0])))


def paired(values,control,min_n=3):
    assert len(values)>=min_n and len(values)==len(control)
    summary=interval([1/v for v in values],[1/v for v in control])
    # Selection uses measured ratios. Only the independent composition lab can
    # promote; neither significance nor A:A is a per-shape selection gate.
    return dict(ratio=math.exp(statistics.mean(map(math.log,values))),
                **summary,wins=summary['R']>1,stage_only=True,
                status='selection evidence; held-out six-load confirmation required')


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
        key=('attention',tuple(row)) if family in ('global','dsm','early','soft') else ('policy',name) if family in ('pdl','boundary') else ('small' if family in ('epi','reuse') else family,row[0]*row[1])
        if key in choices and choices[key][0]['name']==name:
            if list(row) not in choices[key][0]['rows']:choices[key][0]['rows'].append(list(row))
        elif key not in choices or proof['ratio']<choices[key][1]:choices[key]=(spec,proof['ratio'])
    def parse_global():
        global_rows=archive(receipts['global']['log'],'CS_XZ85')
        assert len(global_rows)==4
        for cell in global_rows:
            assert cell['seeds']==16
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
                    candidate('soft_'+kind,(int(match[1]),8,False),dict(ratio=mean,range=[lo,hi],AA_radius=noise,n=3,R=1/mean,CI95=None,selection_range=[1/hi,1/lo],wins=mean<1,stage_only=True))

    def parse_epi():
        epi=receipts['epi']['log']
        assert re.search(r'END seeds=16\b',epi) and not re.search(r'^FAIL ',epi,re.M)
        for m in (1,8,16,32):
            noise_match=re.search(rf'^AA M{m} abs=([\d.]+) us',epi,re.M);assert noise_match
            aa_us=float(noise_match[1])
            kinds=('VEC','PROD','NORM','PAIR')+(('ONCE','ONCE_VEC','ONCE_PAIR') if m!=1 else ())
            for kind in kinds:
                match=re.search(rf'^PAIR M{m} {kind} us=([\d.]+) r=([\d.]+) se=([\d.]+)',epi,re.M);assert match
                us,mean,se=map(float,match.groups());radius=aa_us/max(us/mean,1e-12)
                proof=dict(ratio=mean,se=se,n=16,R=1/mean,CI95=None,CI95_approx=[1/(mean+2.1315*se),1/(mean-2.1315*se)],AA_radius=radius,wins=mean<1,stage_only=True)
                b,q=(1,8) if m==8 else (m,1)
                for w in (True,False):candidate(('reuse_' if kind.startswith('ONCE') else 'epi_')+kind,(b,q,w),proof)

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
                for w in (True,False):candidate('heavy_'+kind,(m//8,8,w),dict(ratio=mean,range=[lo,hi],AA_radius=radius,n=48,R=1/mean,CI95=None,selection_range=[1/hi,1/lo],wins=mean<1,stage_only=True))

    def parse_boundary():
        boundary=decode_boundary(receipts['boundary']['log'])
        assert len(boundary['confirmation']['cells'])==12 and len(boundary['boundaries'])==4,'incomplete boundary archive'
        for family,cells,arms in [('pdl',boundary['confirmation']['cells'],('B','M','D')),('boundary',boundary['boundaries'],('R','P','RP','C','S'))]:
            for kind in arms:
                spec=registry[family+'_'+kind];rows=[];proofs=[]
                for cell in cells:
                    assert len(cell['samples'])==16
                    label=cell['case'];row=(int(re.search(r'\d+',label)[0]),1 if label.startswith('d') else 8,label.endswith('S'))
                    if list(row) not in [list(x) for x in spec['rows']]:continue
                    samples=cell['samples'];aa=[s['A2']/s['A'] for s in samples]
                    proof=paired([s[kind]/s['A'] for s in samples],aa,16)
                    rows.append(row);proofs.append(proof)
                # Rank source-wide alternatives by the six-load geomean,
                # retaining regressions. Unsampled effects remain unconfirmed.
                if rows:
                    measured={r:cell for r,cell in zip(rows,[c for c in cells if (int(re.search(r'\d+',c['case'])[0]),1 if c['case'].startswith('d') else 8,c['case'].endswith('S')) in rows])}
                    seed_times={}
                    for control in boundary['confirmation']['cells']:
                        label=control['case'];row=(int(re.search(r'\d+',label)[0]),1 if label.startswith('d') else 8,label.endswith('S'))
                        sampled=measured.get(row)
                        seed_times[label]=[dict(A=v['A'],A2=v['A2'],B=(sampled['samples'][i][kind] if sampled else v['A'])) for i,v in enumerate(control['samples'])]
                    rr,_=scores(seed_times,'A','B');aa,_=scores(seed_times,'A','A2')
                    summary=interval(rr,aa)
                    evidence.append(dict(arm=family+'_'+kind,**summary,ratio=1/summary['R'],wins=summary['R']>1,
                                         stage_only=True,rows=[list(r) for r in rows],unsampled_effects='unmeasured; confirmation covers all twelve shapes'))
                    if summary['R']>1:choices[('policy',family+'_'+kind)]=(copy.deepcopy(spec),1/summary['R'])

    for name,parse in ((name,fn) for name,fn in [('global',parse_global),('early',parse_early),('dsm',parse_dsm),('soft',parse_soft),('epi',parse_epi),('heavy',parse_heavy),('boundary',parse_boundary)]):
        if name not in receipts:
            evidence.append(dict(batch=name,wins=False,reason='receipt unavailable or official/lab failure; no candidate admitted'))
            continue
        previous=copy.deepcopy(choices)
        try:parse()
        except (AssertionError,KeyError,ValueError,StopIteration,lzma.LZMAError,EOFError) as exc:
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
        keep=max(policies,key=lambda s:next(e['R'] for e in evidence if e.get('arm')==s['name'] and 'R' in e))
        # These edits are source-wide. R/RP also change unmeasured M1/M8 in
        # the component batch; S16 must test those as affected loads, never
        # mislabel them as untouched just because only M16/M32 were sampled.
        keep['rows']=copy.deepcopy(registry[keep['name']]['rows'])
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
        if name=='s15':
            q=json.loads((H.parent/'s15/queue.json').read_text())
            entry=dict(entry,sha256=q['wrapper_sha256'])
        try:receipts[name]=read_receipt(entry,files)
        except AssertionError as exc:
            if name=='s15':raise
            unavailable[name]=str(exc)
    # Exact S15 is the incumbent by owner decision. Never re-gate it or
    # substitute a pruned S15c when the evidence or a report is incomplete.
    assert hashlib.sha256(patches.BASE.read_bytes()).hexdigest()==patches.BASE_SHA
    (R/'problems/muse_decode_layer/cand_r3_best7_s15_final.py').write_bytes(patches.BASE.read_bytes())
    selected,evidence=nominate(receipts)
    import layer_decisions
    layer_selected,layer_evidence,layer_receipts=layer_decisions.nominate(files)
    selected+=layer_selected;evidence+=layer_evidence
    receipts.update({'layer_'+k:v for k,v in layer_receipts.items()})
    assert selected,'no measured selection improvement'
    source,meta=patches.compose(selected)
    out=H/'generated/s16.py';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(source)
    result=dict(meta,source=str(out.relative_to(R)),status='experimental; combined local screen and whole-layer A/B required',
                reset_receipts={k:{name:v for name,v in row.items() if name!='log'} for k,row in receipts.items()},unavailable_receipts=unavailable,evidence=evidence)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(out,meta['sha256'])


if __name__=='__main__':main()
