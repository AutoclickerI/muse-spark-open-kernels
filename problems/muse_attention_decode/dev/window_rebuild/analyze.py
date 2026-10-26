"""Read P5 window receipts without per-load significance cuts; fit the observed cost model."""
import argparse,base64,hashlib,json,lzma,math,statistics
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]

def stats(v):return dict(mean=statistics.mean(v),SE=statistics.stdev(v)/math.sqrt(len(v)))
def unpack(path,marker):
    receipt=json.loads(path.read_text());blob=next(s.split(' ',1)[1] for s in receipt['result']['log_tail'].splitlines() if s.startswith(marker+' '))
    return receipt,json.loads(lzma.decompress(base64.b85decode(blob)))

def diagnostic(path):
    receipt,d=unpack(path,'WINDOW_DIAG_XZ85');cells=[];models=[]
    for r in d['results']:
        assert all(v==len(r['seeds']) for v in r['task'].values())
        arms={n:dict(ratio=stats([a/b for a,b in zip(t,r['times']['0'])]),event_us=stats(t)) for n,t in r['times'].items()}
        phases={}
        for arm in ('0','1','8'):
            nodes=[]
            for n,start,end,*resource in r['profiles'][arm]:
                name=d['names'][n]
                role='scheduler' if 'prepare_varlen' in name else 'combine' if 'Combine<' in name else 'main' if 'device_kernel' in name else name
                nodes.append(dict(role=role,duration_us=(end-start)/1000,start_us=start/1000,end_us=end/1000,grid=resource[:3],threads=resource[3],shared=resource[4],registers=resource[5]))
            phases[arm]=dict(nodes=nodes,span_us=max(n['end_us'] for n in nodes)-min(n['start_us'] for n in nodes))
        cells.append(dict(B=r['B'],window=r['window'],seed_ids=r['seeds'],arms=arms,phases=phases))
    for B in (16,32):
        rows=[r for r in cells if r['B']==B];x=[r['window'] for r in rows];y=[r['arms']['0']['event_us']['mean'] for r in rows]
        xm,ym=statistics.mean(x),statistics.mean(y);slope=sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x);intercept=ym-slope*xm
        models.append(dict(B=B,intercept_including_replay_us=intercept,us_per_key=slope,
            max_abs_residual_us=max(abs(b-intercept-slope*a) for a,b in zip(x,y)),effective_KV_slope_TB_s=B*256/(slope*1e6),
            note='Slope includes gather, tensor math, softmax and scheduling; this is not a DRAM-bandwidth measurement. The intercept also includes replay overhead.'))
    result=dict(job=receipt['id'],receipt=str(path),s15_sha256=d['s15_sha256'],TASK_complete=True,cells=cells,models=models,
        targets=[dict(B=B,unique_KV_bytes=B*(2048+7)*2*64*2,KV_floor_us=B*(2048+7)*2*64*2/4.8e6,target_1_3_us=1.3*B*(2048+7)*2*64*2/4.8e6) for B in (16,32)],
        conclusions=['prepare_v abbreviated symbol is prepare_varlen_num_blocks, not V transpose',
            'five launches at default split; qk prep, scheduler, persistent main, combine, norm/gate',
            'one split removes combine but loses at2048; split tuning does not close the target gap',
            'fuse launches and combine; mainloop work also needs reduction/overlap; copy vs SFU is not isolated by this cost fit'])
    (H/'diagnostic_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def pair(path,output=None):
    receipt,d=unpack(path,'WINDOW_PAIR_XZ85');results=[]
    for r in d['results']:
        n=len(r['seeds']);assert all(v==n for v in r['TASK'].values())
        times=r['times'];arms={};B=int(r['case'][1:-1])
        for name,t in times.items():
            if name=='A':continue
            ratio=stats([x/y for x,y in zip(t,times['A'])]);changed=name!='A2' and not(name=='F8' and B==16)
            arms[name]=dict(**ratio,changed=changed,nominate_to_P4=changed and ratio['mean']<1)
        meta=r['extra']['metadata']
        splits={} if 'values' not in meta else {n:x['values'][:B] for n,x in meta['values'].items()}
        results.append(dict(case=r['case'],seed_ids=r['seeds'],A_A=arms['A2'],arms=arms,actual_dynamic_splits=splits,
                            resources=r['resources'],critical=r['critical'],clocks=r['extra']['clocks']))
        if 'PIPE2' in times:results[-1]['PIPE2_over_RAW2']=stats([x/y for x,y in zip(times['PIPE2'],times['RAW2'])])
    result=dict(job=receipt['id'],receipt=str(path),s15_sha256=d['s15_sha256'],TASK_complete=True,results=results,
        promotion='P4 composes all per-shape point winners and confirms held-out six-load R; no per-load A:A gate')
    (output or H/'pair_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--pair',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.output and not a.pair:p.error('--output applies to --pair')
    r=pair(a.receipt,a.output) if a.pair else diagnostic(a.receipt)
    print('job',r['job'],'TASK complete',r['TASK_complete'])
    if a.pair:
        for row in r['results']:print(row['case'],json.dumps(row['arms']))
    else:
        print(json.dumps(r['models'],indent=2));print(json.dumps(r['targets'],indent=2))
