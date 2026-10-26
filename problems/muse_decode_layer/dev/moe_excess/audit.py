"""Inspect actual sm90a cut kernels, not only source-level guards."""
import hashlib,json,re,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;rows=[];compiler=[];identities={}
for file in ('ladder','ladder_pm'):
 for suffix in ('cu','o','sass','compile.log'):
  p=H/'out'/(file+'.'+suffix);identities[file+'.'+suffix]=hashlib.sha256(p.read_bytes()).hexdigest()
 for part in re.split("ptxas info    : Compiling entry function '",(H/'out'/(file+'.compile.log')).read_text())[1:]:
  name=part.split("'",1)[0];demangled_name=subprocess.check_output(['c++filt',name],text=True).strip()
  if not any(k in demangled_name for k in ('chunk::xkernel','chunk::kernel<4,','mwbr::xgemm','pmz::xphase')):continue
  spill=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',part)
  regs=re.search(r'Used (\d+) registers',part)
  assert spill and regs,demangled_name
  compiler.append(dict(module=file,name=demangled_name,registers=int(regs[1]),stack_bytes=int(spill[1]),spill_store_bytes=int(spill[2]),spill_load_bytes=int(spill[3])))
 s=(H/'out'/(file+'.sass')).read_text();chunks=re.split(r'\n\s*Function : ',s)[1:];names=[c.splitlines()[0].strip() for c in chunks];demangled=subprocess.check_output(['c++filt'],input='\n'.join(names)+'\n',text=True).splitlines()
 for mangled,name,chunk in zip(names,demangled,chunks):
  if not any(k in name for k in ('chunk::xkernel','mwbr::xgemm','pmz::xphase')):continue
  args=name[name.index('<')+1:name.index('>')].split(', ');mode=int(args[-2]);hist=args[-1]=='true';family='heavy' if 'chunk::' in name else 'medium' if 'mwbr::' in name else 'small';first=args[1]=='true'
  counts={key:len(re.findall(pattern,chunk)) for key,pattern in dict(mma=r'\b(?:HGMMA|HMMA)\.',bf16=r'\b(?:HFMA2|HMUL2)[^;\n]*BF16',async_copy=r'\b(?:LDGSTS|UTMA|UTMALDG|UTMABULK)',sink=r'\bSTS\.',global_load=r'\bLDG\.').items()}
  if mode in (0,1,6):assert counts['mma']==0,(name,counts)
  if mode in (2,3,4,5):assert counts['mma']>0,(name,counts)
  if mode==0:assert counts['bf16']==0,(name,counts)
  if mode in (1,2,3,4,5):assert counts['bf16']>0,(name,counts)
  if mode in (5,6):assert counts['async_copy']==0,(name,counts)
  rows.append(dict(module=file,family=family,first=first,mode=mode,hist=hist,counts=counts,name=name))
assert len(rows)>=60,len(rows)
(H/'native_audit.json').write_text(json.dumps(dict(passed=True,arch='sm90a',kernels=rows,compiler_resources=compiler,artifact_sha256=identities,note='MATH/INIT have no global async weight/activation copies; L0/L1 have no MMA; L0 has no BF16 decode products; L1+ retains decode. Spills are disclosed, not filtered: original heavy W2 4/4 bytes; L3 W13 8/8 and W2 12/16; HIST 64/56. Measured clone/sink/observer controls and actual H200 occupancy/resource validation are required for attribution.'),indent=2)+'\n');print('SASS audit',len(rows),'kernels')
