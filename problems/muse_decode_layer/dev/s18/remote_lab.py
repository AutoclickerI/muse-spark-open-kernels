"""The S18-vs-S17 paired lab (build_lab.py's payload, same arms and held-out seeds) hosted by another problem.

    remote_lab.py build                 # payload + wrapper for prefill_host.py (lab-only prefill host)
    remote_lab.py screen [--forced]     # the exact wrapper locally (judge image, stand-ins as screen.py)
    remote_lab.py report <receipt.json> # R/CI with score_gate; a first read, never the freeze

Off the decode host, kernel_fn's globals are not S18, so the payload executes the S18 module itself from its
bundle (the decode wrapper's host hook appended) and draws inputs from that module's copy of TASK's
generate_input. Weights come from TASK's generate_weights at the held-out weight/router seeds, as on decode.
Prepare-phase hosting does not fit: the prefill host's own prepare (114-135 s on the judge) leaves ~150 s of
the 285 s lab budget, below S18's prepare alone (241.6 s). The pair therefore runs at import, as da0cd64b
did, with both candidates in one process so every seed stays paired.
"""
import argparse,base64,hashlib,json,lzma,re,sys,time,types,importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
HOST='muse_prefill_layer';BUDGET=700.0;PAYLOAD=H/'generated/s18_pair_remote.py';WRAPPER=H/'generated/remote_wrapper.py'
SCORED='28b5c189e7b5469c'  # S18's own judge run: the official TASK pass for the host source
def sha(s):return hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()


def decode_body():
    manifest=json.loads((H/'manifest.json').read_text());text=(H/'generated/payload.py').read_text()
    assert sha(text)==manifest['payload_sha256']
    (encoded,)=re.findall(r"_body=lzma\.decompress\(base64\.b85decode\('([^']*)'\)\)",text)
    body=lzma.decompress(base64.b85decode(encoded)).decode()
    assert "hexdigest()=="+repr(sha(body)) in text
    host=(H/'generated/host.py').read_text();packed=(R/manifest['shipped_source']).read_text()
    assert sha(packed)==manifest['shipped_sha256'] and host.startswith(packed)
    return body,host[len(packed):],manifest


def remote_body(body,hook):
    edits=[('started=time.monotonic();base=kernel_fn.__globals__','started=time.monotonic();base=s18_host()'),
           ('generator=_ap_generate_input',"generator=base['_ap_generate_input']"),
           ('    g = kernel_fn.__globals__','    g = S18_HOST'),
           ('scored host unchanged timed code','S18 module executed in-payload on a lab-only '+HOST+' host')]
    for old,new in edits:assert body.count(old)==1,old;body=body.replace(old,new)
    host_fn='''S18_HOST=None
S18_HOST_HOOK=%r
def s18_host():
    # On the decode host this namespace was the host module itself (the packed S18 plus this hook).
    global S18_HOST
    source=S16_BUNDLE['host']+S18_HOST_HOOK
    path='<s18-host-'+hashlib.sha256(source.encode()).hexdigest()[:12]+'>'
    linecache.cache[path]=(len(source),None,source.splitlines(True),path)
    m=types.ModuleType('s18_host');m.__file__=path;exec(compile(source,path,'exec'),m.__dict__)
    if globals().get('S16_LOCAL_INSTALL'):S16_LOCAL_INSTALL(m)
    S18_HOST=m.__dict__;return S18_HOST


'''%hook
    marker='\ndef lab(deadline,weights,state):\n';assert body.count(marker)==1
    body=body.replace(marker,'\n'+host_fn+marker[1:])
    left=[m.start() for m in re.finditer(r"(?<!['\w])kernel_fn\b",body)];assert not left,[body[i-40:i+20] for i in left]
    return body


def build():
    body,hook,manifest=decode_body();body=remote_body(body,hook);encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
    payload='import base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
    payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s18-remote-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
    PAYLOAD.write_text(payload)
    import h200_lab
    wrapper,_=h200_lab.build([(PAYLOAD.stem,sha(payload)[:16],payload)],HOST,phase='import',import_budget=BUDGET,base_file=H/'prefill_host.py')
    assert len(wrapper.encode())<1_000_000;WRAPPER.write_text(wrapper)
    meta=dict(host=HOST,host_file=str((H/'prefill_host.py').relative_to(R)),host_file_sha256=sha((H/'prefill_host.py').read_text()),
              import_budget=BUDGET,payload=str(PAYLOAD.relative_to(R)),payload_sha256=sha(payload),body_sha256=sha(body),
              wrapper_sha256=sha(wrapper),sent_wrapper=f'problems/{HOST}/lab/lab_{PAYLOAD.stem}.py',decode_wrapper_sha256=manifest['wrapper_sha256'],
              sha256=manifest['sha256'],shipped_sha256=manifest['shipped_sha256'],checker_sha256=manifest['checker_sha256'],
              confirmation_seeds=manifest['confirmation_seeds'],arms=list(manifest['arms']),official_TASK_run=SCORED)
    (H/'remote_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))


def screen(forced):
    sp=importlib.util.spec_from_file_location('s18_screen',H/'screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
    meta=json.loads((H/'remote_manifest.json').read_text());source=WRAPPER.read_text();assert sha(source)==meta['wrapper_sha256']
    m=types.ModuleType('submission');m.__file__=str(WRAPPER);lines=[]
    def install():
        m.S16_LOCAL_INSTALL=lambda v:scr.install(v,forced);run=m._lab_run_one
        def checked(name,src,deadline,weights,state):
            out=run(name,src,time.monotonic()+1800,weights,state);lines.extend(out);return out  # sm_120 runs slower
        m._lab_run_one=checked
    m._S18_REMOTE_INSTALL=install;marker='\n_lab_import_start = ';assert source.count(marker)==1
    t0=time.monotonic();exec(compile(source.replace(marker,'\n_S18_REMOTE_INSTALL()'+marker,1),str(WRAPPER),'exec'),m.__dict__)
    import_s=time.monotonic()-t0
    try:m.prepare(None);raise AssertionError('lab-only host prepare returned')
    except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
    sp=importlib.util.spec_from_file_location('s18_report',H/'report.py');rep=importlib.util.module_from_spec(sp);sp.loader.exec_module(rep)
    blob=[l.split(' ',1)[1] for l in lines if l.startswith('S18_XZ85 ')];assert len(blob)==1,lines[-6:]
    raw=rep.decode(json.loads(lzma.decompress(base64.b85decode(blob[0]))))
    mode='forced' if forced else 'normal'
    out=dict(mode=mode,wrapper_sha256=meta['wrapper_sha256'],import_s=round(import_s,1),complete=raw['complete'],failures=raw['failures'],
             task_checks=raw['task_checks'],nearby_task_checks=raw['nearby_task_checks'],agreement_diagnostics=raw['agreement_diagnostics'],
             host_sha256=raw['host_sha256'],seeds=raw['seed_ids'])
    (H/'generated'/f'remote_{mode}_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    assert raw['complete'] and not raw['failures'] and raw['host_sha256']==meta['sha256']


def report(path):
    sp=importlib.util.spec_from_file_location('s18_report',H/'report.py');rep=importlib.util.module_from_spec(sp);sp.loader.exec_module(rep)
    meta=json.loads((H/'remote_manifest.json').read_text());receipt=json.loads(path.read_text())
    assert sha(path.with_suffix('.py').read_bytes())==meta['wrapper_sha256']
    (official,)=R.glob(f"runs/muse_decode_layer/*_{meta['official_TASK_run']}.json");official=json.loads(official.read_text())
    batch=H/'batches'/meta['wrapper_sha256'][:16];batch.mkdir(parents=True,exist_ok=True)
    (batch/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    tail=receipt['result']['log_tail'];lines=[l.split(' ',1)[1] for l in tail.splitlines() if l.startswith('S18_XZ85 ')]
    if not lines:
        result=dict(job=receipt['id'],measured=False,reason='No paired seed archive; not a timing loss',
                    errors=[l for l in tail.splitlines() if l.startswith(('LAB_ERROR','LAB_TRUNCATED','LAB_END'))])
    else:
        (line,)=lines;manifest=json.loads((H/'manifest.json').read_text())
        # The host run fails at prepare by design; S18's TASK pass is its own decode run.
        result=rep.evaluate(dict(id=receipt['id'],result=dict(passed=official['result']['passed'])),
                            rep.decode(json.loads(lzma.decompress(base64.b85decode(line)))),manifest)
        result.update(host=HOST,official_TASK_run=meta['official_TASK_run'],gate_on_this_read=result.pop('freeze_gate'),freeze_gate=False,
                      role='First read on another host; the queued decode check is the confirmation and the only freeze.',
                      lab_end=[l for l in tail.splitlines() if l.startswith('LAB_END')])
    (batch/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('raw','pieces')},indent=2))
    for arm,p in result.get('pieces',{}).items():print(arm,p.get('R'),p.get('CI95'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='cmd',required=True);sub.add_parser('build')
    s=sub.add_parser('screen');s.add_argument('--forced',action='store_true')
    r=sub.add_parser('report');r.add_argument('receipt',type=Path);a=ap.parse_args()
    if a.cmd=='build':build()
    elif a.cmd=='screen':screen(a.forced)
    else:report(a.receipt)
