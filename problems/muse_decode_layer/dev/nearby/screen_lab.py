"""Screen exact shipped payload in local_eval; only Hopper launches substituted."""
import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s15')]
import local_eval as ev


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');a=ap.parse_args()
    mode='forced' if a.forced else 'normal';original=ev.load_module;records=[]
    def loader(name,path):
        m=original(name,path)
        if name=='reference':m.CACHE_CHUNK=8192
        if name=='submission':
            spec=importlib.util.spec_from_file_location('nearby_verify_force',H.parent/'verify_best.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            # Normal mode keeps S15's portable dispatch. Build the FA3 stand-in
            # against an isolated namespace so baseline's ABI can run locally.
            if a.forced:helper.install(m,False)
            else:
                import types
                dummy=types.ModuleType('nearby_baseline_launch')
                dummy.__dict__.update(m.__dict__)
                helper.install(dummy,False)
            m.NEARBY_LOCAL_FLASH=sys.modules['sgl_kernel.flash_attn'].flash_attn_with_kvcache
            import local_standins
            local_standins.install(m)
            fn=m._lab_run_one
            def run(name,source,deadline,weights,state):
                lines=fn(name,source,time.monotonic()+1800,weights,state)
                records.extend(lines);return lines
            m._lab_run_one=run
        return m
    ev.load_module=loader
    target=H/'lab_preview.py'
    sys.argv=['local_eval.py',str(target),'--tests-only','--json',str(H/f'host_{mode}_local_eval.json')]
    try:ev.main()
    finally:
        result=dict(wrapper_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),mode=mode,lines=records)
        for line in records:
            if line.startswith('RESULT '):result['result']=json.loads(line[7:])
        (H/f'host_{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
        assert 'result' in result and result['result']['complete'],result
        assert not result['result']['baseline_error'],result['result']['baseline_error']


if __name__=='__main__':main()
