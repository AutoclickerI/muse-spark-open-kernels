"""Serially screen each repaired layer audit and send its exact bytes as a check."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    for name,priority in (('release',2.9),('library',3.9),('nearby',4.9)):
        rows=json.loads((H/'layer_recovery.json').read_text());entry=rows[name]
        for mode in ('normal','forced'):
            log=H/('recovery_'+name+'_'+mode+'.log')
            with log.open('w') as f:subprocess.run([str(R/'tools/img'),'python',str(H/'screen_repaired.py'),name]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            validation=json.loads((H/'generated/recovery'/name/(mode+'_validation.json')).read_text())
            assert validation['passed'] and validation['sha256']==entry['sha256']
        target=R/entry['file'];assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']
        cmd=[sys.executable,str(R/'tools/okc_run.py'),'send',str(target),'--check','--priority',str(priority),
             '--note','P4 repaired S15 '+name+' layer audit: explicit top-level kernel entry only; original '+entry['original_sha256'][:8]+' payload and gates unchanged; exact normal+forced fullwrapper pass; no result inferred from static rejection.']
        sent=subprocess.run(cmd,cwd=R,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(sent.stdout,end='',flush=True);sent.check_returncode()
        entry.update(screened=True,command=cmd,send_stdout=sent.stdout,priority=priority)
        rows[name]=entry;(H/'layer_recovery.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
