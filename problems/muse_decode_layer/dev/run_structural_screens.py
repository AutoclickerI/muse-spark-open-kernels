"""Serialize GPU screens after the active benchmark's actual process-exit event."""
from pathlib import Path
import os,select,subprocess,sys,psutil,ctypes
p=Path(__file__).resolve().parents[1];root=p.parents[1]
for proc in psutil.process_iter(['pid','cmdline']):
 a=proc.info['cmdline'] or []
 if len(a)>1 and a[1].endswith('/dev/structural_bench.py'):
  try:
   fd=ctypes.CDLL(None,use_errno=True).syscall(434,proc.pid,0)
   if fd>=0:select.select([fd],[],[]);os.close(fd)
  except ProcessLookupError:pass
for name in ('epilogue','chain','chain_occ2','chain_nopf','layer_sweep'):
 logfile=p/f'dev/local_{name}.log'
 with logfile.open('w') as out:
  code=subprocess.run([str(root/'tools/img'),'python',str(root/'tools/local_eval.py'),
   str(p/f'cand_{name}.py'),'--json',str(p/f'dev/local_{name}.json')],cwd=root,
   env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=out,stderr=subprocess.STDOUT).returncode
 print(name,code,flush=True)
 if code:sys.exit(code)
with (p/'dev/graph_inspect.log').open('w') as out:
 code=subprocess.run([str(root/'tools/img'),'python',str(p/'dev/graph_inspect.py')],cwd=root,
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=out,stderr=subprocess.STDOUT).returncode
print('graph_inspect',code,flush=True)
sys.exit(code)
