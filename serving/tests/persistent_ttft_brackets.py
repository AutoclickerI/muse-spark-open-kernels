"""Alternate unchanged harness loads between resident stock/patch servers.

The harness's launch, request timer, workload, graph validation and comparison
functions stay unchanged. This owner holds the processes until all brackets
finish; it does not call run.main(), whose cleanup deliberately kills a server.
"""
import argparse
import hashlib
import http.client
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'serving/harness'))
import compare
import run
from harness_logprob_probe import collect
DATA=run.common.DATA


def resources():
    return {name:subprocess.check_output(['nvidia-smi',query,'--format=csv,noheader'],text=True).splitlines()
            for name,query in [('gpus','--query-gpu=index,memory.used,memory.free,utilization.gpu'),
                               ('processes','--query-compute-apps=pid,gpu_uuid,used_memory')]}


def launch(mode,tp,tree,prefix):
    output=DATA/'results'/f'{prefix}-server-{mode}-tp{tp}';output.mkdir(exist_ok=False)
    target,draft=run.common.model_dirs('margin')
    cfg=argparse.Namespace(target=str(target),draft=str(draft),weights='margin',tp=tp,
        gpus='0' if tp==1 else '0,1',mem_fraction_static=.3,max_total_tokens=300000,
        attention_backend=None,extra=[],env=[])
    if tp==2:cfg.env+=['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0']
    if mode=='portable':
        cfg.env += [f"PYTHONPATH={tree/'python'}:{ROOT/'serving/tests/integration_hook'}:{ROOT/'serving/harness/pyhook'}"]
    port=run.free_port();proc,cmd=run.launch(cfg,tree,output,port)
    server=dict(proc=proc,cmd=cmd,cfg=cfg,tree=tree,out=output,port=port)
    try:
        run.wait_ready(proc,port,output/'server.log',1200)
        capacities=set(re.findall(r'max_total_num_tokens=(\d+)',(output/'server.log').read_text()))
        assert capacities=={'300000'},capacities
    except BaseException:
        run.stop(proc);raise
    return server


def flush(server):
    c=http.client.HTTPConnection('127.0.0.1',server['port'],timeout=60)
    c.request('POST','/flush_cache');r=c.getresponse();body=r.read()
    assert r.status==200,(r.status,body[:200]);c.close()


def workload(server,label,level,convs,turns,*,probe=True):
    output=DATA/'results'/label;output.mkdir(exist_ok=False)
    cfg=server['cfg'];tree=server['tree'];proc=server['proc'];log=server['out']/'server.log'
    items=run.build_workload(0,[level],{level:convs},turns,[2600,5200,18000,3900])
    fingerprint={name:run.common.checkpoint_fingerprint(Path(path)) for name,path in [('target',cfg.target),('draft',cfg.draft)]}
    provenance=run.tree_provenance(tree)
    meta=dict(label=label,provenance=provenance,weights='margin',target=cfg.target,draft=cfg.draft,
        tp=cfg.tp,gpus=cfg.gpus,checkpoint_sha256=fingerprint,
        sampling_params={'temperature':0.0,'top_k':1,'max_new_tokens':220,'ignore_eos':True},
        workload={'seed':0,'levels':[level],'convs':{level:convs},'turns':turns,
                  'first_turn_lens':'2600,5200,18000,3900','requests':len(items),
                  'sha256':hashlib.sha256(json.dumps([r['input_ids'] for r in items]).encode()).hexdigest()},
        server_cmd=server['cmd'],extra_server_args=[],extra_environment=cfg.env,started=time.time())
    (output/'workload.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in items))
    request_file=output/'requests.jsonl';request_file.touch()
    (output/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    before=resources();offset=log.stat().st_size
    def persist(row):
        with request_file.open('a') as out:out.write(json.dumps(row)+'\n')
    try:
        rows=run.run_level(server['port'],level,items,600,persist)
        assert proc.poll() is None
        assert len(rows)==len(items) and all('error' not in r and r['n_out']==220 and len(r['output_ids'])==220 for r in rows)
        traffic=log.read_bytes()[offset:];(output/'traffic.log').write_bytes(traffic)
        traffic_info=run.scan_log(output/'traffic.log')
        assert traffic_info['decode_graph_replayed'] and traffic_info['prefill_graph_replayed'],traffic_info
        if probe:
            collect(run,cfg,tree,output,server['port'])
    except BaseException as exc:
        meta['failure']=repr(exc);raise
    finally:
        (output/'server.log').write_bytes(log.read_bytes())
        meta['log']=run.scan_log(output/'server.log')
        meta['validation_errors']=run.graph_errors(meta['log'],cfg.tp,[])
        meta['provenance_end']=run.tree_provenance(tree)
        meta['summary']=run.summarize([json.loads(s) for s in request_file.read_text().splitlines()],[level])
        (output/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
        (output/'resources_bracket.json').write_text(json.dumps({'before':before,'after':resources()},indent=2)+'\n')
    assert not meta['validation_errors'] and meta['log']['tracebacks']==0,meta
    assert meta['provenance']==meta['provenance_end']
    return label


def paired(stock,patch,tp,level,mode):
    ma,a=compare.load(stock);mb,b=compare.load(patch)
    result=compare.compare(ma,a,mb,b);assert not result['problems'],result['problems']
    assert a.keys()==b.keys() and all(a[k]['output_ids']==b[k]['output_ids'] for k in a)
    av,bv=[r['ttft'] for r in a.values()],[r['ttft'] for r in b.values()]
    row=dict(tp=tp,level=level,mode=mode,stock=stock,patch=patch,exact=len(a),
             p50=statistics.median(bv)/statistics.median(av),p90=compare.quantile(bv,.9)/compare.quantile(av,.9))
    row['margin_pass']=row['p50']<=1.5 and row['p90']<=1.5
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--tree',type=Path,required=True);p.add_argument('--prefix',required=True)
    args=p.parse_args();evidence=DATA/'results'/f'{args.prefix}-evidence';evidence.mkdir(exist_ok=False)
    results=[]
    for tp in (1,2):
        servers={}
        try:
            for mode in ('stock','native','portable'):
                tree=run.common.STOCK_TREE if mode=='stock' else args.tree
                servers[mode]=launch(mode,tp,tree,args.prefix)
                workload(servers[mode],f'{args.prefix}-warm-{mode}-tp{tp}',1,2,3)
            for level,convs in ((1,2),(4,8),(16,16),(32,32)):
                labels={}
                for role in ('stock-a','native','stock-b','portable','stock-c'):
                    mode='stock' if role.startswith('stock') else role
                    server=servers[mode];flush(server)
                    label=f'{args.prefix}-c{level}-{role}-tp{tp}';print('BRACKET',label,flush=True)
                    labels[role]=workload(server,label,level,convs,3)
                for mode,before,after in [('native','stock-a','stock-b'),('portable','stock-b','stock-c')]:
                    for reference in (before,after):
                        row=paired(labels[reference],labels[mode],tp,level,mode)
                        results.append(row);print(json.dumps(row),flush=True)
                (evidence/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        finally:
            for server in servers.values():run.stop(server['proc'])
    print('PERSISTENT AGREEMENT PASS; P50/P90 MARGIN',all(r['margin_pass'] for r in results),flush=True)


if __name__=='__main__':main()
