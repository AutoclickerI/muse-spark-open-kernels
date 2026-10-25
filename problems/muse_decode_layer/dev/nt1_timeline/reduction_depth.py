"""Check the generated FP32 add-chain depth for PAIR and TREE marked reductions."""
import json,re
from pathlib import Path
DEV=Path(__file__).resolve().parent
sass=(DEV/'issue_probe.sass').read_text();out={}
for label,pat in [('PAIR','p_pairILi0ELi0E'),('TREE','p_next0ILi0ELi0E')]:
 name,body=next((n,b) for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if pat in n)
 ops=[]
 for line in body.splitlines():
  m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(.*?)\s*;',line)
  if m:ops.append(m[1])
 marks=[i for i,op in enumerate(ops) if 'CS2R' in op and 'SR_CLOCK' in op];assert len(marks)==2
 regdepth={};maximum=0;fadds=0
 for op in ops[marks[0]+1:marks[1]]:
  regs=re.findall(r'\bR\d+\b',op)
  if not regs:continue
  if re.search(r'\bFADD\b',op):
   depth=1+max([regdepth.get(r,0) for r in regs[1:]] or [0]);maximum=max(maximum,depth);fadds+=1;regdepth[regs[0]]=depth
  elif re.search(r'\b(?:LDS|FMUL|FFMA|MUFU|MOV|S2R|SHF|LOP3|PRMT)\b',op):regdepth[regs[0]]=0
 out[label]={'maximum_dependent_FADD_depth_in_marked_W13':maximum,'FADD':fadds,'caveat':'static contiguous add-chain register dependency only; excludes load/MUFU latency and interleaved scheduling'}
assert out['TREE']['maximum_dependent_FADD_depth_in_marked_W13']<out['PAIR']['maximum_dependent_FADD_depth_in_marked_W13'],out
(DEV/'reduction_depth.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
