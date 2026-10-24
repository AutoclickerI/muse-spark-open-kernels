"""Run the whole intended proj payload through real local portable arithmetic."""
import hashlib
import json
import linecache
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


def main():
    original=local_eval.load_module;report={}
    def loader(name,path):
        module=original(name,path)
        if name=='submission':
            def run(label,source,deadline,weights,state):
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns);start=time.monotonic()
                lines=ns['lab'](min(deadline,start+110.),weights,state);rows=ns['PC_ROWS'];small=ns['PC_SMALL']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),elapsed_s=time.monotonic()-start,rows=rows,small=small,lines=lines,
                    M128_graph_checks=len(rows)*len(ns['PC_NAMES']),small_graph_checks=len(small)*3)
                (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
                assert len(rows)==36 and len(small)==72,lines[-1]
                assert not any('INCOMPLETE' in l or 'REJECT' in l for l in lines),lines
                assert all(e<=.01 and r<=.05 for row in rows+small for e,r in row['errors'])
                assert all(len(row['samples'][0])==12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows+small)
                assert sum(len(l)+1 for l in lines)<=8500
                print('PC_FULL_LOCAL_PASS',report['M128_graph_checks'],'M128 graphs',report['small_graph_checks'],'small graphs',report['elapsed_s'],'s',flush=True)
                return lines
            module._lab_run_one=run
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_bf16_proj/lab/lab_moe_m128_producer_consumer.py'),'--no-baseline','--json',str(HERE/'wrapper_screen.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(report.get('rows',[]))==36
        raise


if __name__=='__main__':main()
