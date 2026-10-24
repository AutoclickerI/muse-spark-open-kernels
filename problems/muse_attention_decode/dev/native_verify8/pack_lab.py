"""Add the native producer measurement to the existing phase payload; keep two payloads total."""
import ast,sys
from pathlib import Path
from build import build
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'problems/muse_attention_decode/dev/producer_phases'))
# Load the phase builder by file to avoid the other build.py on sys.path.
ns={'__file__':str(ROOT/'problems/muse_attention_decode/dev/producer_phases/build.py'),'__name__':'phase_build'}
exec(compile(Path(ns['__file__']).read_text(),ns['__file__'],'exec'),ns)
ns['build']()
build()
p=ROOT/'prof/lab/verify8_producer_phases.py'
source=p.read_text().replace('def lab(deadline,weights):','def original_phase_lab(deadline,weights):')
# Preserve the original gate/clocks while using fewer diagnostic settings to
# reserve compile/runtime for the new producer inside one80-second share.
source=source.replace('ext=phase_ext();phase_ext(True);sms=','ext=phase_ext();sms=')
source=source.replace('((False,(17,3,128)),(True,(17,3,128)),(True,(34,3,64)),(False,(17,4,128)))','((False,(17,3,128)),)')
source+=(HERE/'runtime.py.inc').read_text()+(HERE/'lab.py.inc').read_text().replace('def lab(deadline,weights):','def native_lab(deadline,weights):')
source+='''

def lab(deadline,weights):
    start=time.monotonic()
    # Old phases use a small fixed share; the pair gets its own equal share
    # from h200_lab. No kernel result is read outside this prepare lab.
    first=original_phase_lab(min(deadline,start+35.),weights)
    second=native_lab(deadline,weights)
    return first+second
'''
p.write_text(source)
print(p)
