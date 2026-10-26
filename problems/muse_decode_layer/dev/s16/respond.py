"""Select on frozen S15, screen and send scored, then queue held-out confirmation."""
import argparse,hashlib,json,os,selectors,shutil,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]


def wait_gpu_capacity():
    # Measured packed-screen peak exceeded 20 GiB. Wait only when GPU0 cannot
    # supply the next screen's working set, and wake on a real GPU job exit.
    while True:
        row=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
        uuid,free=row[0].strip(),int(row[1])
        if free>=24576:return
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
        with selectors.DefaultSelector() as selector:
            fds=[]
            for line in procs.splitlines():
                pid,gpu=map(str.strip,line.split(','))
                if gpu!=uuid:continue
                try:fd=os.pidfd_open(int(pid))
                except ProcessLookupError:continue
                selector.register(fd,selectors.EVENT_READ);fds.append(fd)
            if not fds:continue
            print('GPU0 capacity:',free,'MiB free; waiting on running GPU jobs',flush=True)
            selector.select()
            for fd in fds:os.close(fd)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipts',nargs='*',type=Path);ap.add_argument('--screen-only',action='store_true');a=ap.parse_args()
    started=time.monotonic()
    def host(script,*args):subprocess.run([sys.executable,str(H/script),*map(str,args)],cwd=R,check=True)
    host('decide.py',*(['--receipts',*a.receipts] if a.receipts else []))
    selected=json.loads((H/'generated/selection.json').read_text())
    host('build_lab.py');manifest=json.loads((H/'generated/lab_manifest.json').read_text())
    candidate=R/manifest['candidate'];env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    from pack import pack
    shipped=R/'problems/muse_decode_layer'/('cand_r3_best7_s16_'+manifest['sha256'][:12]+'.py');shipped.write_text(pack(candidate.read_text()))
    batch=H/'batches'/manifest['wrapper_sha256'][:16];batch.mkdir(parents=True,exist_ok=True)
    for name in ('selection.json','lab_manifest.json'):shutil.copyfile(H/'generated'/name,batch/name)
    result=dict(candidate=manifest['candidate'],expanded_sha256=manifest['sha256'],shipped_source=str(shipped.relative_to(R)),
                shipped_sha256=hashlib.sha256(shipped.read_bytes()).hexdigest(),wrapper_sha256=manifest['wrapper_sha256'],
                screens={},queued=False,status='experimental S16; S15 frozen until held-out six-load R lower95pct bound>1 and TASK')
    def save():
        result['elapsed_s']=time.monotonic()-started
        for file in (batch/'response.json',H/'response.json'):file.write_text(json.dumps(result,indent=2)+'\n')
    def send(file,mode,priority,note):
        command=[sys.executable,str(R/'tools/okc_run.py'),'send',str(file),mode,'--priority',str(priority),'--note',note]
        sent=subprocess.run(command,cwd=R,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(sent.stdout,end='',flush=True);sent.check_returncode()
        return dict(command=command,stdout=sent.stdout,sender_log='runs/detached/'+hashlib.sha256(file.read_bytes()).hexdigest()[:16]+'.log')
    prior=[json.loads(p.read_text()) for p in (H/'batches').glob('*/response.json')]
    previous=next((p for p in reversed(prior) if p['expanded_sha256']==manifest['sha256'] and p['shipped_sha256']==result['shipped_sha256'] and p.get('scored')),None)
    # Packed source expands to the asserted exact candidate bytes. Screen its
    # real entry points once in each architecture mode, including expanded CUDA.
    for mode in ('normal','forced'):
        out=H/'generated/packed_screen'/mode;out.mkdir(parents=True,exist_ok=True)
        cached=json.loads((out/'local.json').read_text()) if (out/'local.json').exists() else None
        reusable=bool(cached and cached['sha256']==result['shipped_sha256'] and len(cached.get('tests',[]))==13 and len(cached.get('shapes',[]))==12 and all(row['ok'] for k in ('tests','shapes','static') for row in cached[k]))
        if not reusable:
            wait_gpu_capacity()
            with (out/'screen.log').open('w') as log:
                subprocess.run([str(R/'tools/img'),'python',str(H/'screen.py'),str(shipped),'--out',str(out)]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/'local.json').read_text())
        assert data['sha256']==result['shipped_sha256']
        assert len(data['tests'])==13 and len(data['shapes'])==12 and all(row['ok'] for k in ('tests','shapes','static') for row in data[k])
        result['screens'][mode]=dict(tests=13,shapes=12,static=len(data['static']))
        print('PASS packed',mode,flush=True)
    host('audit_composition.py')
    audit=json.loads((H/'generated/expanded_audit.json').read_text());assert audit['passed'] and audit['expanded_sha256']==manifest['sha256']
    result['expanded_audit']=audit
    save()
    if not a.screen_only:
        result['scored']=previous['scored'] if previous else send(shipped,'--scored',-1.1,'P4 S16 scored: frozen S15 60254883 plus global_C128 d1S, DSM C16 v1S/C8 v16S, NORM M1 and PAIR M8/16/32. Shape selections use measured ratios, no per-load A:A gate. Exact packed normal+forced local screens pass. Held-out six-load R confirmation follows.')
        save()
    for mode in ('normal','forced'):
        log=H/'generated'/('lab_'+mode+'.log')
        wait_gpu_capacity()
        with log.open('w') as f:subprocess.run([str(R/'tools/img'),'python',str(H/'screen_lab.py')]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        validation=json.loads((H/'generated'/('lab_'+mode+'_validation.json')).read_text())
        assert validation['complete'] and validation['host_sha256']==manifest['sha256'] and validation['checker_sha256']==manifest['checker_sha256']
        result['screens'][mode].update(paired_seeds=16,paired_shapes=12,task_checks=validation['task_checks'])
        print('PASS confirmation wrapper',mode,flush=True);save()
    if not a.screen_only:
        wrapper=R/manifest['wrapper'];assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==manifest['wrapper_sha256']
        result['confirmation']=send(wrapper,'--check',-1.0,'P4 S16 held-out confirmation: exact screened composition vs frozen S15, each piece off, A:A;16 disjoint input seeds plus fresh weight/router seeds;46/16 weighted six-load R, Student-t95pct CI with A:A noise; unchanged official TASK every arm/seed. No per-load, exact-ID or host-equivalence veto. Import lab620s, private state discarded before real judge prepare.')
        result['queued']=True;save()
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
