"""Full native sm90a opcode/resource evidence for incumbent, ready variants and diagnostic microkernels."""
import collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[3]
def functions(obj):
 s=subprocess.check_output(['cuobjdump','--dump-sass',obj],text=True)
 return re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',s,re.S)
def ops(body):
 result=collections.Counter()
 for line in body.splitlines():
  m=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',line)
  if m:result[m[1].split('.')[0]]+=1
 return dict(result)
rows={}
for tag,obj in [('S14','/tmp/nt1_s14_pipe.o'),*[(t,f'/tmp/nt1_{t}.o') for t in ('pfly','pilp','pk16','plut','pdia','pclk')]]:
 n,b=next((n,b) for n,b in functions(obj) if 'moe_varILi1ELi69E' in n);o=ops(b)
 assert o.get('HMMA',0)==(0 if tag=='pdia' else 16 if tag=='pk16' else 32),(tag,o)
 assert o.get('HMUL2',0)==(0 if tag=='pdia' else 64 if tag=='pk16' else 128),(tag,o)
 if tag=='pdia':
  assert len(re.findall(r'\bLDG\.[^;]*128',b))>=16,(tag,'B-cache loads lost')
  assert len(re.findall(r'\bLDS\.128',b))>=9,(tag,'weight/scales loads lost')
 log=R/f'runs/screen/nt1_{tag}_native.log' if tag!='S14' else R/'runs/screen/nt1_pipe_s14_native.log'
 text=log.read_text();meta=[]
 for nm,stack,st,ld,reg in re.findall(r'Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers',text):
  if 'moe_varILi1ELi69E' in nm:meta=[int(reg),int(stack),int(st),int(ld)]
 assert meta and not meta[2] and not meta[3],(tag,meta)
 source=R/'runs/screen/nt1_pipe_s14_exact/SMALLM_VAR_CUDA.cu' if tag=='S14' else D/(tag+'.cu')
 rows[tag]={'cuda_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'object_sha256':hashlib.sha256(Path(obj).read_bytes()).hexdigest(),'full_function_opcodes':o,'flags69_regs_stack_spillstores_spillloads':meta,'cell_sha256':hashlib.sha256((D/(tag+'_cell.py')).read_bytes()).hexdigest() if tag in ('pfly','pilp','pk16','plut') else None}
cal=[]
for n,b in functions('/tmp/nt1_pipe_micro.o'):
 o=ops(b);args=list(map(int,re.findall(r'Li(\d+)E',n)))
 if 'pipe_work' in n:
  w,g,d,ma,hot,nb,c=args
  if ma==1:assert o.get('HMMA',0)==g*4*nb,(args,o)
  if ma==3:assert o.get('HMMA',0)==g*4,(args,o)
  if ma==0:assert not o.get('HMMA') and not o.get('HMUL2'),(args,o)
  cal.append({'template':args,'opcodes':o})
 if 'pipe_ops' in n:
  op,ilp=args
  target={0:'LOP3',1:'SHF',2:'PRMT',3:'HMUL2',4:'FFMA',5:'HMMA',6:'LDS',7:'LDS',8:'LDS',9:'SHFL',10:'MUFU',13:'LDS'}[op]
  assert o.get(target,0)>=8*ilp,(op,ilp,target,o)
(D/'pipe_native_audit.json').write_text(json.dumps({'ready':rows,'micro_math':cal},indent=2)+'\n')
print('PASS native opcode assertions',len(rows),'images,',len(cal),'math/copy specializations')
for k,v in rows.items():print(k,v['flags69_regs_stack_spillstores_spillloads'],{op:v['full_function_opcodes'].get(op,0) for op in ('LOP3','SHF','PRMT','LDS','HMUL2','HMMA')})
