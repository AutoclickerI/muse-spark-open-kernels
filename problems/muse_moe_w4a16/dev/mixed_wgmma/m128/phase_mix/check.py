"""Force every Hopper dispatch with real portable arithmetic in the host wrapper."""
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
        mod=original(name,path)
        if name=='submission':
            def run(label,source,deadline,weights,state):
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(mod.__dict__);exec(compile(source,filename,'exec'),ns)
                start=time.monotonic();lines=ns['lab'](min(deadline,start+85.),weights,state);rows=ns['PM_ROWS']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows,lines=lines,elapsed_s=time.monotonic()-start,
                    graph_checks=len(rows)*len(ns['PM_NAMES']),portable='actual SM120 immediate MMA; identical native route/bank/pointer glue and finalizer')
                (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
                assert len(rows)==36,lines
                assert not any('INCOMPLETE' in s or 'REJECT' in s for s in lines),lines
                assert all(e<=.01 and r<=.05 for row in rows for e,r in row['errors'])
                assert all(len(row['samples'][0])==12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows)
                assert len([s for s in lines if s.startswith('PH ')])==len(ns['PM_CHOICES'])
                assert sum(len(s)+1 for s in lines)<=8500
                print('PHASE_MIX_LOCAL_PASS',len(rows),'fixtures',report['graph_checks'],'graphs',report['elapsed_s'],'s',flush=True)
                return lines
            mod._lab_run_one=run
        return mod
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_moe_w4a16/lab/lab_moe_m128_phase_mix.py'),'--no-baseline','--json',str(HERE/'wrapper_screen.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(report.get('rows',[]))==36,'payload did not finish'
        raise


if __name__=='__main__':main()
