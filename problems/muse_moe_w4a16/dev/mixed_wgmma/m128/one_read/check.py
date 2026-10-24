"""Validate the sent host wrapper and every expert-wide Hopper launch ABI."""
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
    original_load=local_eval.load_module;report={}
    def loader(name,path):
        module=original_load(name,path)
        if name=='submission':
            def run_one(label,source,deadline,weights,state):
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
                start=time.monotonic();lines=ns['lab'](min(deadline,start+78.),weights,state);rows=ns['OR_ROWS']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows,lines=lines,elapsed_s=time.monotonic()-start,
                    producer_graph_checks=len(rows)*6,NT1_graph_checks=len(rows),portable='SM120 immediate MMA; actual SS shared-A/B layouts, pointer ABI and W2 finalizer')
                (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
                assert len(rows)==36,lines[-1]
                assert not any('INCOMPLETE' in line or 'REJECT' in line for line in lines),lines
                assert all(e<=.01 and r<=.05 for row in rows for e,r in row['errors'])
                assert all(row['expert_blocks']==row['route'][1] for row in rows)
                assert all(len(row['samples'][0])==12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows)
                assert sum(len(line)+1 for line in lines)<=4500
                assert len([line for line in lines if line.startswith('PH ')])==6
                print('ONE_READ_LOCAL_PASS',len(rows),'fixtures;',report['producer_graph_checks'],'producer gates; logical packed-tile fetch factor1.0',flush=True)
                return lines
            module._lab_run_one=run_one
        return module
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_moe_w4a16/lab/lab_moe_m128_no_reread.py'),
        '--no-baseline','--json',str(HERE/'wrapper_screen.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(report.get('rows',[]))==36,'payload failed despite host passing'
        raise


if __name__=='__main__':main()
