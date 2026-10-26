"""Rehearse the exact S21 wrapper on GPU0 with explicit Hopper stand-ins; retain TASK counts."""
import base64,builtins,hashlib,importlib.util,json,lzma,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent
def _load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def main():
    forced='--forced' in sys.argv;scr=_load(H/'screen.py','s21_screen');rep=_load(H/'report.py','s21_report')
    meta=json.loads((H/'manifest.json').read_text());wrapper=H/'generated/wrapper.py';source=wrapper.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==meta['wrapper_sha256']
    m=types.ModuleType('submission');m.__file__=str(wrapper);lines=[];emitted=[]
    def emit(*args,**kwargs):
        emitted.append(kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n'))
        return builtins.print(*args,**kwargs)
    m.print=emit
    def install():
        m.S16_LOCAL_INSTALL=lambda v:scr.install(v,forced);run=m._lab_run_one
        m.S21_LOCAL_ARCHIVE=lambda raw:(H/'generated/rehearsal_raw.json').write_text(json.dumps(raw)+'\n')
        def checked(name,src,deadline,weights,state):
            out=run(name,src,time.monotonic()+3600,weights,state);lines.extend(out);return out  # sm_120 is slower
        m._lab_run_one=checked
    m._S19_INSTALL=install;marker='\n_lab_import_start = ';assert source.count(marker)==1
    t0=time.monotonic();exec(compile(source.replace(marker,'\n_S19_INSTALL()'+marker,1),str(wrapper),'exec'),m.__dict__);import_s=time.monotonic()-t0
    try:m.prepare(None);raise AssertionError('lab-only host prepare returned')
    except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
    sys.path.insert(0,str(H.parents[3]/'tools'))
    from lab_packet import paired_evidence,read_packets
    # Verify the wire the reader will see, including a full reserved error suffix.
    tail=(''.join(emitted)+'P'*4000)[-16000:]
    records=read_packets(tail);assert len(records)==1 and records[0]['source_sha256']==meta['payload_sha256'][:16]
    evidence=paired_evidence(tail,'S21');assert evidence is not None,lines[-8:]
    raw=rep.decode(evidence);mode='forced' if forced else 'normal'
    from promotion import _protocol_arm_valid
    assert all(_protocol_arm_valid(raw,meta,n,len(raw['seed_ids'])) for n in ('A','A2','INC'))
    repeat=raw['original_pair_repeat']
    assert repeat['complete'] and all(v['passed'] for v in repeat['TASK'].values()),repeat
    keep=('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks','agreement_diagnostics','host_prepare_s','build_s','delta_s','attention_plan','host_sha256')
    wire=max(len(x.encode()) for x in emitted if x.startswith('LAB_PACKET '))
    out=dict(mode=mode,wrapper_sha256=meta['wrapper_sha256'],import_s=round(import_s,1),wire_bytes=wire,tail_with_4000_suffix_pass=True,**{k:raw[k] for k in keep})
    out['sampling_protocol']=raw['measurement_protocol'];out['sampling_plan_admitted']=True
    out['telemetry_error']=raw['timing_diagnostics']['telemetry_error']
    out['original_pair_repeat']={k:v for k,v in repeat.items() if k not in ('legacy','symmetric')}
    out['original_pair_repeat']['ratios']={k:repeat[k]['ratios_vs_S20'] for k in ('legacy','symmetric')}
    (H/'generated'/f'lab_{mode}_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    assert raw['complete'] and not raw['failures'] and raw['host_sha256']==meta['sha256']
if __name__=='__main__':main()
