"""Run the exact S19 lab wrapper locally (judge image, GPU0) with S19's stand-ins; record completeness and TASK."""
import base64,hashlib,importlib.util,json,lzma,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent
def _load(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def main():
    forced='--forced' in sys.argv;scr=_load(H/'screen.py','s19_screen');rep=_load(H/'report.py','s19_report')
    meta=json.loads((H/'manifest.json').read_text());wrapper=H/'generated/wrapper.py';source=wrapper.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==meta['wrapper_sha256']
    m=types.ModuleType('submission');m.__file__=str(wrapper);lines=[]
    def install():
        m.S16_LOCAL_INSTALL=lambda v:scr.install(v,forced);run=m._lab_run_one
        def checked(name,src,deadline,weights,state):
            out=run(name,src,time.monotonic()+3600,weights,state);lines.extend(out);return out  # sm_120 is slower
        m._lab_run_one=checked
    m._S19_INSTALL=install;marker='\n_lab_import_start = ';assert source.count(marker)==1
    t0=time.monotonic();exec(compile(source.replace(marker,'\n_S19_INSTALL()'+marker,1),str(wrapper),'exec'),m.__dict__);import_s=time.monotonic()-t0
    try:m.prepare(None);raise AssertionError('lab-only host prepare returned')
    except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
    blob=[l.split(' ',1)[1] for l in lines if l.startswith('S19_XZ85 ')];assert len(blob)==1,lines[-8:]
    raw=rep.decode(json.loads(lzma.decompress(base64.b85decode(blob[0]))));mode='forced' if forced else 'normal'
    keep=('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks','agreement_diagnostics','host_prepare_s','build_s','delta_s','attention_plan','host_sha256')
    out=dict(mode=mode,wrapper_sha256=meta['wrapper_sha256'],import_s=round(import_s,1),blob_chars=len(blob[0]),**{k:raw[k] for k in keep})
    (H/'generated'/f'lab_{mode}_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
    assert raw['complete'] and not raw['failures'] and raw['host_sha256']==meta['sha256']
if __name__=='__main__':main()
