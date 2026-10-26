"""Reproducible live transport proof: build payloads, then verify the canonical judge receipt.

python3 tools/tests/live_loop.py build runs/screen/p1_live
python3 tools/tests/live_loop.py verify <receipt.json> runs/screen/p1_live/manifest.json --out <report.json>
Generated payloads are scratch; runs/<host>/ retains the exact submitted wrapper and receipt.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import h200_lab
from lab_packet import encode_packet,read_packets
from okc_run import queue_metadata

DECISION='verify live maximum packet, optional failure isolation, and PLAN-ranked batching'
NAMES=('p1_maximum','p1_optional','p1_peer')
RESERVATIONS=(8744,1024,512)
COMMON='''import torch,time

def native_pairs():
    seeds=[901,1901]
    pairs=[]
    # Native CUDA work proves the payloads execute on the receipt's actual device.
    # This is a transport probe, never a kernel promotion or correctness claim.
    for seed in seeds:
        gen=torch.Generator(device='cuda');gen.manual_seed(seed)
        x=torch.randn(1024,device='cuda',generator=gen)
        us=[]
        for _ in range(2):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();torch.add(x,x);end.record();end.synchronize()
            us.append(round(start.elapsed_time(end)*1000,4))
        pairs.append(dict(seed=seed,old_us=us[0],new_us=us[1]))
    return dict(identities=dict(candidate='torch.add transport probe',incumbent='torch.add transport probe'),
                seeds=seeds,paired=pairs,checks=dict(gpu=torch.cuda.get_device_name(),
                cc=list(torch.cuda.get_device_capability()),TASK='not applicable: transport probe'),complete=True)
'''
MAXIMUM=COMMON+'''
import base64,hashlib,json,random,string,zlib

def lab(deadline,weights,state):
    packet=native_pairs()
    packet['padding']=''
    packet['padding_sha256']=''
    rng=random.Random(20261026)
    data=''.join(rng.choices(string.ascii_letters+string.digits,k=20000))
    def fit(n):
        packet['padding']=data[:n]
        packet['padding_sha256']=hashlib.sha256(packet['padding'].encode()).hexdigest()
        raw=json.dumps(packet,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
        blob=base64.b85encode(zlib.compress(raw,9)).decode()
        return len('LAB_PACKET p1_maximum '+'0'*16+' '+hashlib.sha256(raw).hexdigest()+' '+blob+'\\n')
    lo,hi=0,len(data)
    while lo<hi:
        mid=(lo+hi+1)//2
        if fit(mid)<=8744:lo=mid
        else:hi=mid-1
    for n in range(max(0,lo-128),min(len(data),lo+128)+1):
        if fit(n)==8744:break
    else:raise RuntimeError('could not fill exact maximum packet allocation')
    print('P1_NOISY_IMPORT '+'x'*18000,flush=True)
    return dict(packet=packet,logs=['P1 maximum packet 8744 bytes','diagnostics '+'x'*18000])
'''
OPTIONAL='''def lab(deadline,weights,state):
    raise RuntimeError('P1 expected isolated optional-arm failure; no device fault')
'''
PEER=COMMON+'''
def lab(deadline,weights,state):
    return dict(packet=native_pairs(),logs=['P1 peer ran after the failed optional payload'])
'''


def build(directory,lane=None):
    directory.mkdir(parents=True,exist_ok=True)
    paths=[]
    for name,reservation,body in zip(NAMES,RESERVATIONS,(MAXIMUM,OPTIONAL,PEER)):
        resources=dict(compile_s=0,run_s=3,output_bytes=reservation,context_poison='none')
        path=directory/(name+'.py')
        path.write_text('LAB_RESOURCES='+repr(resources)+'\n'+body)
        paths.append(path)
    embedded=[(p.stem,hashlib.sha256(p.read_bytes()).hexdigest()[:16],p.read_text()) for p in paths]
    batches=h200_lab.plan_batches(embedded,budget_s=30,output_budget=h200_lab.LAB_ONLY_LIMIT)
    assert len(batches)==1 and len(batches[0])==3
    source,_=h200_lab.build(batches[0],'muse_bf16_proj',lab_only=True,import_budget=30.0)
    (directory/'wrapper.py').write_text(source)
    manifest=dict(host='muse_bf16_proj',names=NAMES,reservations=dict(zip(NAMES,RESERVATIONS)),
                  payload_sha256={p.stem:hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in paths},
                  wrapper_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  planner_batches=[len(b) for b in batches],output_budget=h200_lab.LAB_ONLY_LIMIT,
                  queue=queue_metadata(lane,DECISION),expected_maximum_wire_bytes=8744)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


def verify(receipt_path,manifest_path,out):
    receipt=json.loads(receipt_path.read_text());manifest=json.loads(manifest_path.read_text())
    assert hashlib.sha256(receipt_path.with_suffix('.py').read_bytes()).hexdigest()==manifest['wrapper_sha256']
    assert receipt['result']['gpu']=='H200'
    assert receipt['mode'] in ('check','test'),receipt['mode']
    tail=receipt['result']['log_tail']
    records=read_packets(tail)  # raises if any present checksum or compressed data is invalid
    assert [r['name'] for r in records]==list(NAMES)
    assert all('LAB_BEGIN '+name+' ' in tail for name in NAMES)
    details=[]
    for record in records:
        name=record['name'];packet=record['packet']
        assert record['source_sha256']==manifest['payload_sha256'][name]
        wire=encode_packet(name,record['source_sha256'],packet)
        assert wire in tail
        assert len(wire.encode())+1 <= manifest['reservations'][name]
        details.append(dict(name=name,checksum=record['checksum'],checksum_valid=True,
                            complete=packet['complete'],wire_bytes=len(wire.encode())+1))
        if name=='p1_optional':
            assert packet['complete'] is False
            assert 'P1 expected isolated optional-arm failure' in '\n'.join(packet['lines'])
        else:
            assert packet['complete'] is True
            assert packet['checks']['cc']==[9,0] and 'H200' in packet['checks']['gpu']
            assert packet['seeds']==[901,1901] and len(packet['paired'])==2
            assert all(row['old_us']>0 and row['new_us']>0 for row in packet['paired'])
            if name=='p1_maximum':
                assert details[-1]['wire_bytes']==manifest['expected_maximum_wire_bytes']
                assert hashlib.sha256(packet['padding'].encode()).hexdigest()==packet['padding_sha256']
    rows=[json.loads(l) for l in (ROOT/'runs/ledger.jsonl').read_text().splitlines()]
    (row,)=[r for r in rows if r['job']==receipt['id']]
    for key in ('lane','decision','priority','priority_override'):
        assert row[key]==manifest['queue'][key],(key,row.get(key),manifest['queue'][key])
    suffix=tail.rsplit('LAB_END ',1)[1].split('\n',1)[1]
    report=dict(job=receipt['id'],receipt=str(receipt_path.resolve().relative_to(ROOT)),gpu=receipt['result']['gpu'],
                host=manifest['host'],planner_batches=manifest['planner_batches'],packets=details,
                isolation='primary and subsequent peer complete; only deliberate optional payload incomplete',
                queue={k:row[k] for k in ('lane','decision','priority','priority_override')},
                log_tail_chars=len(tail),suffix_after_packets_chars=len(suffix),
                expected_lab_only_host_stop=receipt['result'].get('passed') is False,
                wrapper_sha256=manifest['wrapper_sha256'])
    assert 'lab-only host' in tail  # the expected prepare stop, separate from per-payload completeness
    if out:out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('build');p.add_argument('directory',type=Path);p.add_argument('--lane',help='active PLAN lane; unlabeled if omitted after P1 closes')
    p=sub.add_parser('verify');p.add_argument('receipt',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.cmd=='build':build(args.directory,args.lane)
    else:verify(args.receipt,args.manifest,args.out)
