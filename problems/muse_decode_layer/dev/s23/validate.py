"""S23 local screens: static, normal/forced stand-in screens, exact packed rehearsal, lab transport."""
import json,os,subprocess,sys
import compose
H=compose.H;R=compose.R;ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
SCREEN='problems/muse_bf16_proj/dev/public_ports/night/screen_s22.py'

def run(cmd,log):
 with (H/log).open('w') as out:p=subprocess.run(cmd,cwd=R,env=ENV,stdout=out,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{cmd} failed; see {H.relative_to(R)}/{log}')

def main():
 m=json.loads((H/'manifest.json').read_text());c=json.loads((H/'contracts.json').read_text());candidate=m['shipped_source']
 assert c['passed'] and c['source_sha256']==m['sha256']
 run([sys.executable,'tools/local_eval.py',candidate,'--tier','static','--json',str(H/'generated/static.json')],'static.log')
 for mode in ('normal','forced'):
  run(['tools/img','python',SCREEN,candidate,'--out',str(H/'screen'/mode)]+(['--forced'] if mode=='forced' else []),'screen_'+mode+'.log')
 run(['tools/img','python',str(H/'screen_lab.py')],'rehearsal.log')
 run(['tools/img','python','tools/local_eval.py',str(H/'generated/wrapper.py'),'--tier','wrapper','--transport','lab-only','--json',str(H/'generated/transport.json')],'transport.log')
 lab=json.loads((H/'generated/lab_forced_validation.json').read_text());assert lab['source_sha256']==m['sha256']
 v=dict(source_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],contracts=len(c['checks']),packed=lab,transport=json.loads((H/'generated/transport.json').read_text()))
 for mode in ('normal','forced'):
  x=json.loads((H/'screen'/mode/'local.json').read_text());assert x['sha256']==m['shipped_sha256'] and all(t['ok'] for t in x['tests']) and len(x['shapes'])==12
  v[mode]=dict(TASK=len(x['tests']),graphs=len(x['shapes']),passed=True)
 (H/'validation.json').write_text(json.dumps(v,indent=2)+'\n');(H/'stages'/f"p{m['phase']}"/'validation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k] for k in ('contracts','normal','forced')}),{k:lab[k] for k in ('TASK','nearby','wire_bytes','import_s','host_prepare_s')})
if __name__=='__main__':main()
