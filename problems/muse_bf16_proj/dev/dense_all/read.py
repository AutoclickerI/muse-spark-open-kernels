"""Read only retained H200 archives; preserve raw paired samples and all failures."""
import argparse,csv,hashlib,json,math,re,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
protocol={};exec((H/'protocol.py.inc').read_text(),protocol)
def ci(values):
    z=[math.log(x) for x in values];mu=statistics.mean(z);critical=2.364624251 if len(z)==8 else 2.131449546
    half=critical*statistics.stdev(z)/math.sqrt(len(z));return [math.exp(mu),math.exp(mu-half),math.exp(mu+half)]
def active(t,role):
    if role in ('o','pool_router','unpooler'):return ['sliding','full']
    if role=='qkv_gate':return [] if t==1 else ['full'] if t==16 else ['sliding'] if t==32 else ['sliding','full']
    if role in ('qkv3','gate_apply'):return ['sliding','full'] if t==1 else ['sliding'] if t==16 else ['full'] if t==32 else []
    return []
def analyse(receipt):
    d=json.loads(receipt.read_text());tail=d.get('result',{}).get('log_tail','');lines=result_lines(tail);match=re.search(r'^DATA (\S+)','\n'.join(lines),re.M)
    if not match:
        bad=dict(job=d['id'],valid=False,receipt=str(receipt.relative_to(R)),reason='No retained dense archive; not a measured loss',errors=[x for x in lines if 'ERROR' in x or 'Error' in x or 'TRUNCATED' in x]);(H/'results.json').write_text(json.dumps(bad,indent=2)+'\n');print(bad);return bad
    raw=protocol['da_decode'](match[1]);transport_errors=[error for packet in read_packets(tail) for error in packet['packet'].get('transport_errors',[])];rows=[];primary=[]
    for cell in raw['records']:
        vals=[[v*raw['unit_us'] for v in x] for x in cell['values']];a=vals[0];aa=ci([b/v for v,b in zip(a,vals[1])]);cellrows=[]
        for i,arm in enumerate(cell['arms'][2:],2):
            ratios=[b/v for v,b in zip(a,vals[i])];ratio=ci(ratios)
            row=dict(t=cell['t'],role=cell['role'],kind=cell['kind'],arm=arm,ratio=ratio,AA_ratio=aa,A_us=statistics.mean(a),B_us=statistics.mean(vals[i]),control_samples=a,candidate_samples=vals[i],AA_samples=vals[1],active_at=active(cell['t'],cell['role']),point_win=ratio[0]<1)
            if 'control' in cell:row['control']=cell['control']
            if 'code_bytes' in cell:row.update(code_bytes=cell['code_bytes'][arm],issued_weight_byte_bound=cell['issued_weight_byte_bound'][arm])
            if 'agreement' in cell:row['agreement_diagnostic']=cell['agreement']
            rows.append(row);cellrows.append(row)
        if cell['kind']=='neighbour' or cell['role'] in ('qkv_gate','qkv3'):
            wins=[x for x in cellrows if x['point_win'] and x['active_at']]
            if wins:primary.append(min(wins,key=lambda x:x['ratio'][0]))
    result=dict(job=d['id'],receipt=str(receipt.relative_to(R)),receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),valid=raw['complete']==36 and raw['failure'] is None and not transport_errors,transport_errors=transport_errors,complete=raw['complete'],failure=raw['failure'],rows=rows,primary_stage_selection=primary,base_prepare_s=raw['base_prepare_s'],added_prepare=raw['added_prepare'],seeds=raw['seeds'],elapsed_s=raw['elapsed_s'],CI='Student-t7 over eight paired log ratios; candidate/control time; selection data, not G0',limits=['One weight/router seed','Producer and actual-neighbour timings are separate; chain rows select O/pool/unpool','Raw gate torch.mm and inactive QKV3/full-QKV cells are diagnostics','G0 and additional production prepare cost need the selected bank implementation'])
    (H/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (H/'ratios.csv').open('w',newline='') as f:
        fields=['t','role','kind','arm','A_us','B_us','ratio','AA_ratio','active_at','point_win','code_bytes','issued_weight_byte_bound'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
    for row in primary:print(row['t'],row['role'],row['kind'],row['arm'],*[round(x,6) for x in row['ratio']])
    print('VALID',result['valid'],'complete',raw['complete'],'failure',raw['failure']);print('PREPARE',raw['base_prepare_s'],raw['added_prepare'])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt');a=p.parse_args();path=Path(a.receipt)
    if not path.exists():
        found=list((R/'runs').glob('muse_*/*'+a.receipt+'.json'));assert len(found)==1,found;path=found[0]
    analyse(path.resolve())
