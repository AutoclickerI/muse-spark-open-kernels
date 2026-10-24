"""Run the full synthetic MoE sweep in the actual projection host wrapper."""
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
    old=local_eval.load_module;report={}
    def loader(name,path):
        m=old(name,path)
        if name=='submission':
            def run(label,source,deadline,weights,state):
                filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
                ns=dict(m.__dict__);exec(compile(source,filename,'exec'),ns);started=time.monotonic()
                lines=ns['lab'](min(deadline,started+105.),weights,state);rows=ns['SW_ROWS'];phases=ns['SW_PHASES']
                report.update(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),elapsed_s=time.monotonic()-started,rows=rows,phases=phases,lines=lines,
                    graph_checks=len(rows)*len(ns['SW_NAMES']),phase_seed_count=len(set(p['seed'] for p in phases)),choices=ns['SW_CHOICES'])
                (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
                assert len(rows)==36 and report['phase_seed_count']==12,lines[-1]
                assert all(e<=.01 and r<=.05 for row in rows for e,r in row['errors'])
                assert all(len(row['samples'][0])==12 and all(len(v)==6 for v in row['samples'][1:]) for row in rows)
                assert not any('INCOMPLETE' in l or 'REJECT' in l for l in lines),lines
                assert sum(len(l)+1 for l in lines)<=8800
                print('SWEEP_LOCAL_PASS',len(rows),'fixtures',report['graph_checks'],'graphs',report['phase_seed_count'],'phase seeds',report['elapsed_s'],'s',flush=True)
                return lines
            m._lab_run_one=run
        return m
    local_eval.load_module=loader
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_bf16_proj/lab/lab_moe_m128_mix_sweep.py'),'--no-baseline','--json',str(HERE/'wrapper_screen.json')]
    try:local_eval.main()
    except SystemExit as exc:
        if exc.code==0:assert len(report.get('rows',[]))==36
        raise


if __name__=='__main__':main()
