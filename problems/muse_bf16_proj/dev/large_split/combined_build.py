"""One prepared S19 state for the repaired oracle and structural candidate batch."""
import ast,base64,hashlib,json,lzma
from pathlib import Path
import build,ss_build
H=Path(__file__).resolve().parent;R=H.parents[3]
source=build.base+'\n'+build.body+'\n'+ss_build.body
oracle=(H/'lab.py.inc').read_text().replace('def lab(deadline,weights,state=None):','def oracle_lab(deadline,weights,state=None):')
needle="prepare_s=time.monotonic()-t0";assert oracle.count(needle)==1
oracle=oracle.replace(needle,needle+"\n    globals()['LS_SHARED']=dict(m=m,weights=weights,state=st,base_prepare_s=base_prepare)")
# Every native entry gets a bounded eager preflight before capture.
needle="labels=['S19'";a=oracle.index(needle);oracle=oracle[:a]+oracle[a:]
needle="graphs=[capture(lambda label=label:value(label),x,label.endswith('hotX')) for label in labels]";assert oracle.count(needle)==1
oracle=oracle.replace(needle,"\n                for mode in m['LS_MODES']:\n                    try:m['ls_run'](x,role,bank,mode,pads[mode]);torch.cuda.synchronize()\n                    except Exception as e:raise RuntimeError('oracle native preflight '+str((t,role,mode))+': '+repr(e)) from e\n                "+needle)
ss=(H/'ss_lab.py.inc').read_text().replace('def lab(deadline,weights,state=None):','def candidate_lab(deadline,weights,state=None):')
a=ss.index('    weights=task[');b=ss.index(';bank=m[\'ss_prepare\'](st)',a)
ss=ss[:a]+"    weights=LS_SHARED['weights'];st=LS_SHARED['state'];base_prepare=LS_SHARED['base_prepare_s']"+ss[b:]
# ss_load itself is overridden below; all functions and CUDA sources are in the one module.
trailer='''
def ss_load():return LS_SHARED['m']
def lab(deadline,weights,state=None):
    data={};started=time.monotonic()
    globals()['LS_LOCAL_ARCHIVE']=lambda r:data.update(oracle=r)
    globals()['SS_LOCAL_ARCHIVE']=lambda r:data.update(candidate=r)
    try:
        oracle_lab(deadline,weights,state)
        if data.get('oracle',{}).get('complete'):candidate_lab(deadline,weights,state)
    except Exception as error:
        import traceback
        data['failure']=repr(error)+' | '+' | '.join(traceback.format_exc().splitlines()[-6:])
    data['complete']=all(data.get(k,{}).get('complete') for k in ('oracle','candidate')) and not data.get('failure')
    data['elapsed_s']=time.monotonic()-started
    if globals().get('COMBINED_LOCAL_ARCHIVE'):COMBINED_LOCAL_ARCHIVE(data)
    # Keep every paired timing; avoid duplicating narrative and per-arm error
    # summaries already enforced by TASK assertions in the canonical source.
    data.get('oracle',{}).pop('limits',None)
    for row in data.get('candidate',{}).get('rows',[]):
        row['max_error']=max(row.pop('errors',[]) or [0.])
        row['max_router_error']=max(row.pop('router_errors',[]) or [0.])
    data=combined_metadata(data);encoded=combined_encode(data);assert combined_decode(encoded)==data
    assert len(encoded)<7800,len(encoded)
    return ['META P10 repaired oracle+shared-weight candidate;one S19 prepare;all arms8seeds cold;original FP32router TASK;no synthetic promotions','COMBINED_XZ85 '+encoded,'END complete='+str(data['complete'])]
'''
payload='LAB_RESOURCES={"compile_s":260,"run_s":260,"output_bytes":8744,"context_poison":"device"}\nimport torch,triton,triton.language as tl,time,hashlib,statistics,json,base64,lzma,linecache,types\nLS_BASE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\nSS_BASE=LS_BASE\nLS_CHECK='+repr(build.checker)+'\nSS_CHECK=LS_CHECK\nLS_TASK='+repr(build.tasks)+'\nSS_TASK=LS_TASK\n'+(H/'codec.py.inc').read_text()+'\n'+oracle+'\n'+ss+'\n'+trailer
compile(payload,'<P10 joint mechanism/candidate>','exec');(H/'generated/combined_payload.py').write_text(payload);(H/'generated/combined_source.py').write_text(source)
(H/'combined_manifest.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),bytes=len(payload.encode()),oracle_seed_ids=[15000319+104729*i for i in range(8)],candidate_seed_ids=[16000319+104729*i for i in range(8)],shared_weight_seed=15000311,shared_router_seed=15000313,base_sha256=hashlib.sha256(build.base.encode()).hexdigest()),indent=2)+'\n');print(len(payload))
