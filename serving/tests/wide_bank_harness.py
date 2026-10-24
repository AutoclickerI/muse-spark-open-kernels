"""Read-only harness with real-width expert fixtures and the actual reader."""
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'serving/harness'))
import common
import run

tp=int(sys.argv[sys.argv.index('--tp')+1]) if '--tp' in sys.argv else 1
for kind in ('margin','random'):
    common.WEIGHT_SETS[kind]=(f'muse-bank-width-tp{tp}-{kind}',f'dflash-bank-width-{kind}')
base_launch,base_ready=run.launch,run.wait_ready
base_stop=run.stop
reference=None
if '--agreement-reference' in sys.argv:
    i=sys.argv.index('--agreement-reference');reference=sys.argv[i+1];del sys.argv[i:i+2]


def launch(args,tree,out,port):
    args.mem_fraction_static=.3;args.max_total_tokens=300000
    if args.tp==2:args.env+=['SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0']
    if tree!=common.STOCK_TREE:
        args.env += [f"PYTHONPATH={tree/'python'}:{ROOT/'serving/tests/shared_reader_hook'}:{ROOT/'serving/harness/pyhook'}"]
    proc,cmd=base_launch(args,tree,out,port)
    proc.wide_context=(args,tree,out,port,cmd)
    return proc,cmd


def ready(proc,port,log,timeout):
    elapsed=base_ready(proc,port,log,timeout)
    assert set(re.findall(r'max_total_num_tokens=(\d+)',Path(log).read_text()))=={'300000'}
    return elapsed


run.launch,run.wait_ready=launch,ready


def stop(proc):
    try:
        args,tree,out,port,cmd=proc.wide_context
        if args.weights=='random' and proc.harness_fired and proc.poll() is None:
            import teacher_forced_workload as teacher
            ref=reference or out
            path,items,requests=teacher.load_run(ref)
            records=[]
            for item in items:
                forced=list(requests[item['key']]['output_ids'])
                assert len(forced)==220
                prompt=item['input_ids'];response=teacher.request(port,prompt+forced,600)
                raw=response.get('meta_info',{}).get('input_token_logprobs',[])
                assert [r[1] for r in raw]==prompt+forced
                logprobs=teacher.finite_logprobs(response,len(prompt),220)
                records.append(dict(key=item['key'],level=item['level'],conv=item['conv'],turn=item['turn'],
                                    prompt_len=len(prompt),forced_ids=forced,logprobs=logprobs))
            dest=out/'forced';dest.mkdir(exist_ok=False)
            (dest/'forced.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (dest/'meta.json').write_text(json.dumps(dict(label=out.name+'-forced',tree=str(tree),
                    reference=str(path),requests=len(records),command=cmd),indent=2))
            print('FULL 220-POSITION TEACHER FORCE',len(records),flush=True)
    finally:base_stop(proc)


run.stop=stop
if __name__=='__main__':
    if '--forced' in sys.argv:
        sys.argv.remove('--forced')
        import teacher_forced_workload
        teacher_forced_workload.main()
    else:run.main()
