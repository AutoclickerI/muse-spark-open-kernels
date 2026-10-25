"""Local five-family compound rehearsal; no receipt substitution or remote send."""
import json,os,subprocess,sys,time
from pathlib import Path
import patches
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    begin=time.monotonic()
    names=['early_E4','soft_mix25','epi_PAIR','heavy_DIRECT_NORMAL','pdl_M']
    selection=H/'generated/rehearsal_selection.json'
    selection.write_text(json.dumps(dict(patches=[patches.registry()[name] for name in names],rehearsal=True,performance_claim=False),indent=2)+'\n')
    subprocess.run([sys.executable,str(H/'build_lab.py'),'--selection',str(selection)],cwd=R,check=True)
    result=dict(rehearsal=True,performance_claim=False,remote_send=False,patches=names,screens={})
    manifest=json.loads((H/'generated/lab_manifest.json').read_text());env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'}
    from pack import pack
    packed=H/'generated/rehearsal_packed.py';packed.write_text(pack((R/manifest['candidate']).read_text()))
    for mode in ('normal','forced'):
        out=H/'generated/rehearsal_screen'/mode;out.mkdir(exist_ok=True,parents=True)
        with (out/'screen.log').open('w') as f:
            subprocess.run([str(R/'tools/img'),'python',str(H/'screen.py'),str(R/manifest['candidate']),'--out',str(out)]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/'local.json').read_text());assert data['sha256']==manifest['sha256']
        assert len(data['tests'])==13 and len(data['shapes'])==12 and all(r['ok'] for k in ('tests','shapes','static') for r in data[k])
        packed_out=H/'generated/rehearsal_packed_screen'/mode;packed_out.mkdir(exist_ok=True,parents=True)
        with (packed_out/'screen.log').open('w') as f:
            subprocess.run([str(R/'tools/img'),'python',str(H/'screen.py'),str(packed),'--out',str(packed_out)]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        shipped=json.loads((packed_out/'local.json').read_text())
        assert len(shipped['tests'])==13 and len(shipped['shapes'])==12 and all(r['ok'] for k in ('tests','shapes','static') for r in shipped[k])
        with (H/'generated'/('rehearsal_lab_'+mode+'.log')).open('w') as f:
            subprocess.run([str(R/'tools/img'),'python',str(H/'screen_lab.py')]+(['--forced'] if mode=='forced' else []),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        lab=json.loads((H/'generated'/('lab_'+mode+'_validation.json')).read_text());assert lab['complete'] and lab['checker_sha256']==manifest['checker_sha256']
        result['screens'][mode]=dict(tests=13,graph_shapes=12,static=len(data['static']),packed_tests=13,packed_shapes=12,paired_shapes=12,seeds=16,task_checks=lab['task_checks'],failures=lab['failures'])
        print('REHEARSAL_PASS',mode,result['screens'][mode],flush=True)
    result.update(elapsed_s=time.monotonic()-begin,expanded_sha256=manifest['sha256'],wrapper_sha256=manifest['wrapper_sha256'],passed=True)
    (H/'rehearsal.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
