"""One event-driven receipt watcher for this active review; no polling timer."""
import contextlib
import ctypes
import io
import os
from pathlib import Path
import subprocess
import sys
import report

root=report.ROOT;libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
for directory in ('runs','runs/pending','runs/waiting','runs/muse_attention_decode','runs/muse_moe_w4a16','runs/muse_prefill_layer'):
 if libc.inotify_add_watch(fd,os.fsencode(root/directory),0x8|0x80|0x200)<0:raise OSError(ctypes.get_errno(),directory)
readers={};last=None
while True:
 with contextlib.redirect_stdout(io.StringIO()) as output:states=report.main()
 summary=output.getvalue()
 if summary!=last:print(summary,flush=True);last=summary
 for batch,q in states.items():
  if q['state'] in ('measured','unmeasured') and q['job'] not in readers:
   log=Path('/tmp')/('p3b_review_'+q['job']+'.log')
   # Every service read goes through the authorized lab wrapper.
   with log.open('w') as stream:
    readers[q['job']]=subprocess.Popen([sys.executable,str(root/'tools/h200_lab.py'),'read',q['job']],stdout=stream,stderr=subprocess.STDOUT,cwd=root)
   print('READ',batch,q['job'],str(log),flush=True)
 if all(q['state'] in ('measured','unmeasured') for q in states.values()):break
 os.read(fd,262144)  # blocks until a queue/receipt event
os.close(fd)
for p in readers.values():p.wait()
