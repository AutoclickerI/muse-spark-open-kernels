"""Read sm90a resources and opcode counts for the exact round2 embedded CUDA images."""
import hashlib,json,re,subprocess
from pathlib import Path
from build_epi_next import KINDS
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
rows={}
for kind,tag in KINDS.items():
 log=(ROOT/f'runs/screen/nt1_{tag}_r2_sm90.log').read_text();assert 'error:' not in log
 entries=[]
 for name,stack,store,load,regs in re.findall(r'Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',log):
  f=re.search(r'moe_varILi(\d+)ELi(\d+)E',name);entries.append(dict(nt=int(f[1]),flags=int(f[2]),registers=int(regs),stack=int(stack),spill_stores=int(store),spill_loads=int(load)))
 sass=subprocess.check_output(['cuobjdump','--dump-sass',f'/tmp/nt1_{tag}_r2.o'],text=True)
 fn,body=next((n,b) for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if 'moe_varILi1ELi69E' in n)
 counts={op:len(re.findall(r'\b'+op+r'\.',body)) for op in ('HMUL2','HMMA')};assert counts=={'HMUL2':128,'HMMA':32},counts
 assert len(entries)==5 and all(not x[k] for x in entries for k in ('stack','spill_stores','spill_loads'))
 fp32_atoms=len(re.findall(r'\b(?:ATOMG|REDG)[^;]*?\.F32\.',body))
 if kind in ('local','four_local'):assert fp32_atoms==0,(kind,fp32_atoms)
 else:assert fp32_atoms>0,(kind,fp32_atoms)
 rows[kind]={'norm_FP32_global_atomic_static_sites':fp32_atoms,'cell_sha256':hashlib.sha256((DEV/(tag+'_cell.py')).read_bytes()).hexdigest(),'sm90a':entries,'flags69_full_SASS':counts}
(DEV/'epi_next_resources.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
