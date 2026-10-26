"""Full unchanged S15n/e/L payload gates on an explicit-entry repaired wrapper."""
import argparse,base64,hashlib,importlib.util,json,lzma,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s15')]
import local_eval as ev


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('name',choices=('nearby','release','library'));ap.add_argument('--forced',action='store_true');a=ap.parse_args()
    mode='forced' if a.forced else 'normal';entry=json.loads((H/'layer_recovery.json').read_text())[a.name]
    target=R/entry['file'];assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']
    force=load(H.parent/'verify_best.py','p4_recovered_force');standins=load(H.parent/('library_chain/library_standins.py' if a.name=='library' else 's15/local_standins.py'),'p4_recovered_standins')
    original=ev.load_module;records=[]
    def loader(name,path):
        m=original(name,path)
        if name=='reference':m.CACHE_CHUNK=8192
        if name=='submission':
            if a.forced:force.install(m,False)
            else:
                dummy=types.ModuleType('p4_baseline_flash');dummy.__dict__.update(m.__dict__);force.install(dummy,False)
            m.NEARBY_LOCAL_FLASH=sys.modules['sgl_kernel.flash_attn'].flash_attn_with_kvcache
            standins.install(m)
            def install(v):
                if a.forced:force.install(v,False)
                standins.install(v)
            m.NEARBY_LOCAL_INSTALL=install;m.RELEASE_LOCAL_INSTALL=install
            run=m._lab_run_one
            def checked(name,source,deadline,weights,state):
                lines=run(name,source,time.monotonic()+1800,weights,state);records.extend(lines);return lines
            m._lab_run_one=checked
        return m
    ev.load_module=loader;out=H/'generated/recovery'/a.name;out.mkdir(parents=True,exist_ok=True)
    sys.argv=['local_eval.py',str(target),'--tests-only','--json',str(out/(mode+'_local.json'))]
    try:ev.main()
    except SystemExit as exc:
        if exc.code==0:
            values=[json.loads(line[7:]) for line in records if line.startswith('RESULT ')]
            values += [json.loads(lzma.decompress(base64.b85decode(line[12:]))) for line in records if line.startswith('RESULT_XZ85 ')]
            assert len(values)==1,records[-2:]
            result=values[0];assert result['complete'],result
            if a.name=='nearby':assert not result['baseline_error'] and all(len(row)==7 and row[-1] for row in result['rows'])
            else:assert not result['errors'] and not result['failures']
            (out/(mode+'_validation.json')).write_text(json.dumps(dict(sha256=entry['sha256'],passed=True,result=result),indent=2)+'\n')
        raise


if __name__=='__main__':main()
