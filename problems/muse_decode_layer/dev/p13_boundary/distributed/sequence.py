"""Bounded event-driven receipt reader: BOUND first, route only after it finishes.

No advice posting, polling cadence, shared edits or automatic commits. The
ordinary h200_lab/okc_run sender owns quota/reset waits. This reader waits on
that exact sender's pidfd, validates the immutable receipt, then sends step2.
"""
import argparse
import ctypes
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path.insert(0,str(H))
from report import report

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def state(**values):
    values['reader_pid']=os.getpid();values['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    path=H/'out/sequence_status.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(values,indent=2)+'\n');tmp.replace(path)

def one(kind,host):
    manifest=json.loads((H/(kind+'_lab_manifest.json')).read_text())
    payload=H/'out'/('p13_'+kind+'.py');screen=json.loads((H/(kind+'_screen.json')).read_text())
    assert sha(payload)==manifest['payload_sha256']==screen['payload_sha256']
    source=f'problems/{host}/lab/lab_p13_{kind}.py'
    decision='distributed BOUND scale identity in production graph' if kind=='boundary' else 'parallel per-row route in production graph'
    note=f'P13 {kind} on exact {manifest["base_series"]} {manifest["base_sha256"][:8]}; INC/A/A2/B/B2,8fixedfreshseeds x10complete symmetric all-round cohorts; all12S/F and exact busiest shard/bothsoft;480primary+54TASK/nearby local checks, sm90a changed compile and full checksum/tail roundtrip pass. No native claim until receipt. Distributed rows, unchanged weight bits and FP32 residual.'
    # Watch before starting the detached sender. Its first marker can appear
    # after h200_lab returns; wait for the actual filesystem event, not a timer.
    libc=ctypes.CDLL(None,use_errno=True);notify=libc.inotify_init1(os.O_CLOEXEC|os.O_NONBLOCK)
    if notify<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    for folder in ('waiting','pending',host):
        directory=R/'runs'/folder;directory.mkdir(exist_ok=True)
        if libc.inotify_add_watch(notify,os.fsencode(directory),0x00000100|0x00000080|0x00000008)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
    try:
        subprocess.run([sys.executable,str(H.parent/'send.py'),'send',str(payload.relative_to(R)),
                        '--host',host,'--lab-only','--phase','import','--import-budget','820','--lane','P13','--decision',decision,'--note',note],cwd=R,check=True)
        expected=sha(R/source)
        while True:
            waiting=[];pending=[];finished=[]
            for folder,dest in (('waiting',waiting),('pending',pending),(host,finished)):
                for p in (R/'runs'/folder).glob('*.json'):
                    try:d=json.loads(p.read_text())
                    except json.JSONDecodeError:continue  # a CLOSE_WRITE event follows
                    if folder==host:
                        if p.with_suffix('.py').exists() and sha(p.with_suffix('.py'))==expected:dest.append(d)
                    elif d.get('file')==source and d.get('lane')=='P13':dest.append(d)
            if waiting or pending or finished:break
            select.select([notify],[],[]);os.read(notify,65536)
    finally:os.close(notify)
    assert len(waiting)<=1 and len(pending)<=1,(waiting,pending)
    if waiting:
        entry=waiting[0];assert entry['sha256']==expected
        (H/(kind+'_queue.json')).write_text(json.dumps(dict(entry,payload_sha256=manifest['payload_sha256'],scope='Unsubmitted while quota is empty; native R unmeasured'),indent=2)+'\n')
        state(phase=kind+'_sender_wait',sender_pid=entry['pid'],source=source,wrapper_sha256=expected,payload_sha256=manifest['payload_sha256'])
        # This host Python lacks os.pidfd_open; use the tested Linux x86_64
        # syscall directly. Waiting on the fd still observes actual completion.
        assert os.uname().machine=='x86_64'
        libc=ctypes.CDLL(None,use_errno=True);fd=libc.syscall(434,entry['pid'],0)
        if fd<0:
            err=ctypes.get_errno()
            if err==3:fd=None  # completion won the race; receipt must exist below
            else:raise OSError(err,'pidfd_open')
        if fd is not None:
            try:select.select([fd],[],[])
            finally:os.close(fd)
    elif pending:
        state(phase=kind+'_native_wait',job=pending[0]['job'],source=source,wrapper_sha256=expected)
        subprocess.run([sys.executable,str(R/'tools/okc_run.py'),'collect',pending[0]['job']],cwd=R,check=True)
    matches=[]
    for path in (R/'runs'/host).glob('*.json'):
        # Receipt JSONs carry the judge's result, not our wrapper hash; the .py beside it is the exact source sent.
        source_copy=path.with_suffix('.py')
        if source_copy.exists() and sha(source_copy)==expected:matches.append(path)
    assert len(matches)==1,('sender ended without one matching immutable receipt',kind,matches)
    result=report(matches[0],kind)
    state(phase=kind+'_complete',job=result['job'],lab_winners=result['lab_winners'])
    return result

def run(host,kinds):
    try:
        results={kind:one(kind,host) for kind in kinds}
        state(phase='complete',**{kind:v['job'] for kind,v in results.items()},**{kind+'_winners':v['lab_winners'] for kind,v in results.items()})
    except BaseException as exc:
        state(phase='stopped_with_error',error=type(exc).__name__+':'+str(exc));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--host',default='muse_prefill_layer');parser.add_argument('--kinds',nargs='+',choices=('boundary','route'),default=['boundary','route']);parser.add_argument('--detach',action='store_true');args=parser.parse_args()
    if args.detach:
        (H/'out').mkdir(exist_ok=True)
        previous=H/'out/sequence_status.json'
        if previous.exists():
            old=json.loads(previous.read_text());pid=old.get('reader_pid')
            if pid:
                try:os.kill(pid,0)
                except ProcessLookupError:pass
                else:raise RuntimeError('P13 receipt reader already running: '+str(pid))
        with (H/'out/sequence.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--host',args.host,'--kinds',*args.kinds],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('P13 bounded receipt reader PID',child.pid,'status',H/'out/sequence_status.json')
    else:run(args.host,args.kinds)
