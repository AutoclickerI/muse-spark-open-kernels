"""One command after the reset receipts land: select, build, screen, then queue S16.

The existing S15 gate must pass first. This script never alters the reset queue.
It stops on a missing receipt, failed screen or incomplete paired wrapper.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--receipts',nargs='*',type=Path);ap.add_argument('--screen-only',action='store_true');a=ap.parse_args()
    started=time.monotonic()
    def host(script,*args):subprocess.run([sys.executable,str(H/script),*map(str,args)],cwd=R,check=True)
    host('decide.py',*(['--receipts',*a.receipts] if a.receipts else []))
    host('build_lab.py')
    manifest=json.loads((H/'generated/lab_manifest.json').read_text())
    candidate=R/manifest['candidate'];env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    from pack import pack
    shipped=R/'problems/muse_decode_layer/cand_r3_best7_s16.py';shipped.write_text(pack(candidate.read_text()))
    screens={}
    for mode in ('normal','forced'):
        out=H/'generated/combined_screen'/mode;out.mkdir(parents=True,exist_ok=True)
        cmd=[str(R/'tools/img'),'python',str(H/'screen.py'),str(candidate),'--out',str(out)]+(['--forced'] if mode=='forced' else [])
        with (out/'screen.log').open('w') as log:subprocess.run(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/'local.json').read_text());assert data['sha256']==manifest['sha256']
        assert len(data['tests'])==13 and len(data['shapes'])==12 and all(row['ok'] for k in ('tests','shapes','static') for row in data[k])
        log=H/'generated'/('lab_'+mode+'.log')
        with log.open('w') as f:subprocess.run([str(R/'tools/img'),'python',str(H/'screen_lab.py')]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        validation=json.loads((H/'generated'/('lab_'+mode+'_validation.json')).read_text())
        assert validation['complete'] and validation['host_sha256']==manifest['sha256']
        screens[mode]=dict(tests=13,shapes=12,static=len(data['static']),paired_seeds=16,paired_shapes=12)
    result=dict(candidate=manifest['candidate'],expanded_sha256=manifest['sha256'],shipped_source=str(shipped.relative_to(R)),shipped_sha256=hashlib.sha256(shipped.read_bytes()).hexdigest(),wrapper_sha256=manifest['wrapper_sha256'],screens=screens,
                elapsed_s=time.monotonic()-started,queued=False,status='screened experimental; freeze waits on paired S16 receipt')
    if not a.screen_only:
        command=[sys.executable,str(R/'tools/h200_lab.py'),'send',str(R/manifest['payload']),'--host','muse_decode_layer',
                 '--base-file',str(R/manifest['host_file']),'--priority','-0.9','--check',
                 '--note','P4 S16 second-round reset winners on admitted S15; all retained pieces off,S15 back,A:A,16 fresh seeds,all12 shapes,normal+forced per-arm/composed/wrapper pass; candidates selected from exact reset receipts.']
        subprocess.run(command,cwd=R,check=True)
        wrapper=R/manifest['wrapper'];assert hashlib.sha256(wrapper.read_bytes()).hexdigest()==manifest['wrapper_sha256']
        result['queued']=True;result['command']=command
    (H/'response.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
