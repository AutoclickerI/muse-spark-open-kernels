"""Wait on the completion event of an existing task, then run one dependent command."""
import ctypes,os,select,sys,psutil
needle=sys.argv[1]
for p in psutil.process_iter(['pid','cmdline']):
 a=p.info['cmdline'] or []
 if len(a)>1 and a[1].endswith(needle):
  fd=ctypes.CDLL(None,use_errno=True).syscall(434,p.pid,0)
  if fd>=0:select.select([fd],[],[]);os.close(fd)
os.execvp(sys.argv[2],sys.argv[2:])
