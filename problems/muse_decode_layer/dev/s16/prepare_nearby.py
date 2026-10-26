"""Prepare guarded nearby dispatch on the exact queued S16; never send it."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
import capture,patches
from respond import wait_gpu_capacity
H=Path(__file__).resolve().parent;R=H.parents[3]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--screen',action='store_true');a=ap.parse_args()
    selection=H/'batches/504e1fa66298bf9e/selection.json'
    specs=json.loads(selection.read_text())['patches']
    base,base_meta=patches.compose(specs)
    assert base_meta['sha256']=='435df5f78a0d157f979b4b472865a139a28ac333fe275b1aebc0055a76dbf6d3'
    source,meta=patches.compose(specs+['layer_nearby'])
    old=(H.parents[1]/'cand_r3_best7_s15n.py').read_text()
    guarded=(H.parent/'nearby/native_validation/guarded_s15n.py').read_text()
    assert hashlib.sha256(old.encode()).hexdigest().startswith('ebceda91')
    assert hashlib.sha256(guarded.encode()).hexdigest()=='9cc90d2e4764f0b47016ec526aa9742a86394ec87c3e838bf66c6376a672bfc8'
    source=patches.apply_delta(source,capture.delta(old,guarded))
    target=H/'generated/nearby_on_s16.py';target.write_text(source)
    result=dict(base_sha256=base_meta['sha256'],guarded_s15n_sha256=hashlib.sha256(guarded.encode()).hexdigest(),
                sha256=hashlib.sha256(source.encode()).hexdigest(),source=str(target.relative_to(R)),screens={},sent=False,
                decision='Native nearby TASK pass + S16 held-out R pass: use this composition after its screens. If S16 loses, retain S15 and use the guarded S15n handoff instead; no inference from missing/failed controls.')
    for mode in ('normal','forced') if a.screen else ():
        wait_gpu_capacity();out=H/'screen/nearby_composed'/mode;out.mkdir(parents=True,exist_ok=True)
        with (out/'screen.log').open('w') as log:
            subprocess.run([str(R/'tools/img'),'python',str(H/'screen_nearby.py'),str(target),'--out',str(out)]+(['--forced'] if mode=='forced' else []),
                           cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
        validation=json.loads((out/'validation.json').read_text());assert validation['passed'] and validation['sha256']==result['sha256']
        result['screens'][mode]=dict(cases=validation['cases'],eager=True,graph_replays=2,passed=True,
                                    report=str((out/'validation.json').relative_to(R)))
        (H/'nearby_composition.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS nearby composition',mode,result['sha256'][:8],flush=True)
    (H/'nearby_composition.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
