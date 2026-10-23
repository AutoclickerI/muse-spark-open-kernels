"""Development orchestration only: official local_eval screens for frozen candidates."""
import subprocess,sys
from pathlib import Path
root=Path('problems/muse_prefill_layer')
for filename in ('cand_attention_fp8.py','cand_attention_fp8_contig.py','cand_attention_fp8_contig256.py','cand_canonical_attention.py'):
  stem=Path(filename).stem
  log=root/'dev'/f'local_{stem}_final.log'
  with log.open('w') as f:
    result=subprocess.run([sys.executable,'tools/local_eval.py',str(root/filename),'--tests-only','--json',str(root/'dev'/f'local_{stem}_final.json')],stdout=f,stderr=subprocess.STDOUT)
  print(filename,'exit',result.returncode,'log',log,flush=True)
  if result.returncode: raise SystemExit(result.returncode)
