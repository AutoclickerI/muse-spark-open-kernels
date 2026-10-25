"""Require exact source identities and complete local evidence before send."""
import hashlib
import json
from pathlib import Path
import build
H=Path(__file__).resolve().parent


def main():
    candidate=H.parents[1]/'cand_r3_best7_s15L.py'
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert candidate.read_text()==build.compose()
    result=dict(base_sha256=build.SHA,candidate_sha256=digest,candidate={},lab={})
    for mode in ('normal','forced'):
        data=json.loads((H/f's15L_{mode}.json').read_text())
        checks=json.loads((H/f's15L_{mode}_local_eval.json').read_text())
        assert data['candidate_sha256']==digest and data['complete']
        assert len(data['tests'])==143 and all(t['eager_ok'] and t['graph_ok'] for t in data['tests'])
        assert len(checks['static'])==25 and all(t['ok'] for t in checks['static'])
        result['candidate'][mode]=dict(fixtures=143,eager_checks=143,replay_checks=143,static=25)
        host=json.loads((H/f'host_{mode}.json').read_text())
        hc=json.loads((H/f'host_{mode}_local_eval.json').read_text());r=host['result']
        assert r['complete'] and not r['errors'] and not r['failures']
        assert len(r['shapes'])==12 and len(r['loads'])==6 and r['task_checks']==768
        assert all(len(row['us'])==8 and row['n']==8 for row in r['shapes'].values())
        assert len(hc['tests'])==13 and all(t['ok'] for t in hc['tests'])
        assert len(hc['static'])==2 and all(t['ok'] for t in hc['static'])
        wrapper=hashlib.sha256((H/'out/lab_preview.py').read_bytes()).hexdigest()
        assert host['wrapper_sha256']==wrapper
        proof=r['native_sources_unchanged']
        assert proof['L']['nf_rows']==[[128,'o'],[256,'o']]
        assert proof['L']['vdsk_rows']==[[128,'pool_router'],[128,'unpooler']]
        assert proof['L']['vq8_rows']==[[1,8,False],[32,8,True]]
        assert not proof['no_o']['nf_rows']
        assert [128,'pool_router'] not in proof['no_pool128']['vdsk_rows']
        assert [128,'unpooler'] not in proof['no_unpool128']['vdsk_rows']
        assert [1,8,False] not in proof['no_v1full']['vq8_rows']
        assert [32,8,True] not in proof['no_v32slide']['vq8_rows']
        result['lab'][mode]=dict(wrapper_sha256=wrapper,shapes=12,loads=6,seeds=8,replays=8,arms=8,
            task_checks=768,host_tests=13,static=2,elapsed_s=r['elapsed'])
    native=json.loads((H/'native_compile.json').read_text())
    assert len(native)==4 and all(x['ok'] for x in native)
    proof=json.loads((H/'preservation.json').read_text())
    assert proof['qkv_vdsk_binary_identical'] and proof['candidate_sha256']==digest
    result.update(native_sm90a_compile=4,preservation='36 literal identities and unchanged qkv VDSK SASS',
        limitations=['native Hopper runtime, concurrency and performance pending reset',
                     'sm120 local stand-ins for Hopper-only attention, NF wide O and oversized VDSK launch'])
    (H/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
