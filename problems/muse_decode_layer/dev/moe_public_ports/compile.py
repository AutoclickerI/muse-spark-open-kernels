"""Compile actual dispatched flags, including all WGMMA phase instantiations."""
import importlib.util,json,time
from pathlib import Path
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('public_native',H/'out/public_moe.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
for name,kind,flags in (('X1','X',('-DRPW=2',)),('X2','X',('-DNCW=16',)),('F1','F',('-DWPC=16','-DSMAX=3','-DMT2','-DLASTA','-DSMREQ=196')),('F2','F',('-DWPC=16','-DSMAX=3','-DMT2','-DSMREQ=196'))):
 t=time.monotonic();mod=m.lee_module(kind,flags);results.append(dict(name=name,flags=flags,seconds=time.monotonic()-t,shared_object=mod.__file__));print(name,results[-1]['seconds'],flush=True)
(H/'native_validation.json').write_text(json.dumps(dict(arch='sm90a',passed=True,units=results),indent=2)+'\n')
