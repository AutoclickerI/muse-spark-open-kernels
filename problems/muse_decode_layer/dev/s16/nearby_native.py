"""Retarget the admitted 67-shape TASK lab to the exact S16 nearby merge."""
import argparse,ast,base64,difflib,hashlib,json,lzma,os,select,subprocess,sys
from pathlib import Path
import patches
from respond import process_exit_fd,wait_gpu_capacity
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'generated/nearby_native'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--send',action='store_true');ap.add_argument('--wait-pid',type=int);a=ap.parse_args();O.mkdir(parents=True,exist_ok=True)
    if a.wait_pid:
        try:fd=process_exit_fd(a.wait_pid)
        except ProcessLookupError:pass
        else:
            print('Waiting on own running audit',a.wait_pid,flush=True);select.select([fd],[],[]);os.close(fd)
    receipt=json.loads(next((R/'runs/muse_attention_decode').glob('*_c03a3ae8ddfd490d.json')).read_text());assert receipt['result']['passed']
    import decide
    proof=decide.archive(receipt['result']['log_tail'],'NATIVE_TASK_XZ85');assert proof['complete'] and proof['task_checks']==268 and not proof['failures'] and not proof['errors']
    frozen=json.loads((H/'frozen.json').read_text());folder=next(p.parent for p in (H/'batches').glob('*/lab_manifest.json') if json.loads(p.read_text())['sha256']==frozen['expanded_sha256'])
    manifest=json.loads((folder/'lab_manifest.json').read_text());specs=json.loads((folder/'selection.json').read_text())['patches'];base,meta=patches.compose(specs)
    if manifest.get('prepare_routing_revision'):
        import prepare_constraints
        base=prepare_constraints.apply(base)
    meta['sha256']=hashlib.sha256(base.encode()).hexdigest();assert meta['sha256']==frozen['expanded_sha256']
    near=(H/'generated/nearby_on_s16.py').read_text();digest=hashlib.sha256(near.encode()).hexdigest();composition=json.loads((H/'nearby_composition.json').read_text());assert digest==composition['sha256'] and composition['base_sha256']==meta['sha256']
    template=(H.parent/'nearby/native_validation/payload.py').read_text();template=template[:template.index('\ndef lab(deadline,weights,state):')]
    lines=template.splitlines(True)
    for node in reversed(ast.parse(template).body):
        if isinstance(node,ast.Assign) and any(getattr(t,'id','') in ('VALIDATION_BASE','S15N_SHA','S15N_EDITS') for t in node.targets):
            lines[node.lineno-1:node.end_lineno]=[]
    template=''.join(lines)
    sys.path.insert(0,str(R/'tools'));import local_eval
    official=local_eval.tests_of((R/'problems/muse_decode_layer/TASK.md').read_text());assert len(official)==13
    template+='\nCASES += '+repr([dict(c,label='official-'+str(i)) for i,c in enumerate(official)])+'\n'
    shapes=80;expected=shapes*4
    before,after=base.splitlines(True),near.splitlines(True)
    edits=[(lo,hi,''.join(after[c:d])) for op,lo,hi,c,d in difflib.SequenceMatcher(None,before,after,autojunk=False).get_opcodes() if op!='equal']
    text=template+'\nVALIDATION_BASE='+repr(base)+'\nS15N_SHA='+repr(digest)+'\nS15N_EDITS='+repr(edits)+'\n'
    text+=(H.parent/'nearby/native_validation.py.inc').read_text().replace("'S15'","'S16'").replace("'S15n'","'S16n'")
    text=text.replace('native S15/guarded S15n correctness','native frozen S16/guarded S16 nearby correctness').replace('65audit+2capacity shapes','65audit+2capacity+13official shapes')
    body=text;encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    text='import base64,lzma,linecache\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n_f="<s16-native-nearby>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_native_lab=lab\ndef lab(deadline,weights,state):return _native_lab(deadline,weights,state)\n'
    payload=O/'s16_nearby_native.py';payload.write_text(text)
    import h200_lab
    from pack import pack
    shipped=O/'candidate.py';shipped.write_text(pack(near))
    raw,_=h200_lab.build([(payload.stem,hashlib.sha256(text.encode()).hexdigest()[:16],text)],'muse_decode_layer',base_file=shipped,phase='import',import_budget=550)
    assert len(raw.encode())<1_000_000,len(raw.encode())
    wrapper=O/'wrapper.py';wrapper.write_text(raw);sha=hashlib.sha256(raw.encode()).hexdigest()
    result=dict(base_sha256=meta['sha256'],candidate_sha256=digest,shipped_sha256=hashlib.sha256(shipped.read_bytes()).hexdigest(),wrapper_sha256=sha,shapes=shapes,task_checks=expected,screens={},sent=False)
    for mode in ('normal','forced'):
        wait_gpu_capacity();out=O/mode;out.mkdir(exist_ok=True)
        prefix='\ndef NEARBY_LOCAL_INSTALL(m):\n import sys\n sys.path.insert(0,'+repr(str(H))+')\n import screen\n screen.install(m,'+repr(mode=='forced')+')\nNEARBY_LOCAL_INSTALL(__import__("sys").modules[__name__])\n'
        mark='import time as _lab_time\n';assert raw.count(mark)==1
        local=raw.replace(mark,prefix+mark).replace('_lab_import_start + 550','_lab_import_start + 1800')
        mark='    return fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)';assert local.count(mark)==1
        target=out/'lines.json'
        local=local.replace(mark,'    result=fn(deadline, weights, state) if len(_inspect.signature(fn).parameters) >= 3 else fn(deadline, weights)\n    __import__("pathlib").Path('+repr(str(target))+').write_text(__import__("json").dumps(result))\n    return result')
        file=out/'local.py';file.write_text(local)
        with (out/'screen.log').open('w') as log:subprocess.run([str(R/'tools/img'),'python','tools/local_eval.py',str(file),'--tests-only','--json',str(out/'local.json')],cwd=R,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0'},stdout=log,stderr=subprocess.STDOUT,check=True)
        lines=json.loads(target.read_text());r=decide.archive('\n'.join(lines),'NATIVE_TASK_XZ85');assert r['complete'] and r['task_checks']==expected and not r['failures'] and not r['errors']
        result['screens'][mode]=dict(TASK_checks=expected,passed=True);print('PASS exact nearby-native wrapper',mode,flush=True)
    if a.send:
        p=subprocess.run([sys.executable,str(R/'tools/okc_run.py'),'send',str(wrapper),'--check','--priority','-0.9','--note','P4 current frozen S16 '+meta['sha256'][:8]+' plus guarded nearby '+digest[:8]+':80shapes incl13official and B64/B65,2seeds,320 native public TASK gates,2 captured replays. Actual packed candidate is decode host; normal+forced full wrapper pass. No timing claim; scored dispatch unchanged. Standalone9cc90d2e passed c03a3ae8.'],cwd=R,text=True,stdout=subprocess.PIPE,check=True)
        result.update(sent=True,stdout=p.stdout);print(p.stdout,end='',flush=True)
    (H/'nearby_native_queue.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
