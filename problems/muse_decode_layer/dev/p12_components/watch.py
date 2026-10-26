"""Wait for these five queued runs by filesystem events, then use h200_lab read."""
import ctypes,os,json,subprocess,sys,select
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
targets={x['sha256']:('P12_'+x['role']) for x in json.loads((H/'queue.json').read_text())}
targets.update({'1a03dfa6906c8242f25c8d808aec8ef96ab6de960f905257195369842bf021dd':'P10_pool','d50be2e7d31519d68877a08613c11ab0bdac83caa74086708ca177382dae914b':'P10_all'})
lib=ctypes.CDLL(None,use_errno=True);fd=lib.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
if fd<0:raise OSError(ctypes.get_errno(),'inotify init')
for p in (R/'runs/pending',R/'runs'):
    assert lib.inotify_add_watch(fd,os.fsencode(p),0x00000008|0x00000080|0x00000100)>=0
found={};done=set();children={}
while len(done)<len(targets):
    ledger=[json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines() if l.strip()]
    for row in ledger:
        if row['sha256'] in targets:found[row['sha256']]=row
    for p in (R/'runs/pending').glob('*.json'):
        try:row=json.loads(p.read_text())
        except (ValueError,FileNotFoundError):continue
        if row['sha256'] in targets:found[row['sha256']]=row
    for sha,row in found.items():
        if sha not in children:
            label=targets[sha];log=(H/(label+'_read.log')).open('w')
            child=subprocess.Popen([sys.executable,str(R/'tools/h200_lab.py'),'read',row['job']],stdout=log,stderr=subprocess.STDOUT,cwd=R)
            children[sha]=(child,log);print('ACCEPTED',label,row['job'],flush=True)
        if sha not in done and any(x['job']==row['job'] for x in ledger):
            done.add(sha);print('RECEIPT',targets[sha],row['job'],flush=True)
    if len(done)==len(targets):break
    select.select([fd],[],[])
    try:os.read(fd,65536)
    except BlockingIOError:pass
for child,log in children.values():child.wait();log.close()
os.close(fd);print('ALL_RECEIPTS',flush=True)
