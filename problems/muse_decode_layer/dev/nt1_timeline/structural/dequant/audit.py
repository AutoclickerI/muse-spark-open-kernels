"""Count exact sm90a decoder images and preserve hashes/resources before H200."""
import ast,collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];m=json.loads((D/'cells.json').read_text());rows={}
for name,spec in m.items():
 cell=(R/spec['cell']).read_text();assert hashlib.sha256(cell.encode()).hexdigest()==spec['cell_sha256']
 tree=ast.parse(cell);cu=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'))
 assert hashlib.sha256(cu.encode()).hexdigest()==spec['cuda_sha256']
 path=R/'runs/screen/p3a_structural'/(name+'.cu');path.write_text(cu);obj=path.with_suffix('.o')
 p=subprocess.run(['nvcc','-O3','-std=c++20','-arch=sm_90a','-c',str(path),'-o',str(obj),'-Xptxas=-v'],capture_output=True,text=True);path.with_suffix('.compile.log').write_text(p.stdout+p.stderr);p.check_returncode()
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);path.with_suffix('.sass').write_text(sass)
 fn,b=next((fn,b) for fn,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if re.search(r'moe_varILi1ELi69E',fn))
 ops=collections.Counter(k.split('.')[0] for k in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][\w.]*)',b))
 rr=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers.*?(\d+) bytes smem',p.stderr);assert rr,(name,p.stderr)
 stack,stores,loads,regs,static=map(int,rr.groups());assert not stores and not loads and static==65536 and ops['HMMA']==32 and ops['HMUL2']==128
 rows[name]=dict(cuda_sha256=spec['cuda_sha256'],cell_sha256=spec['cell_sha256'],registers=regs,stack_bytes=stack,spill_stores=stores,spill_loads=loads,shared_static_bytes=static,full_flags69_ops=dict(ops))
 print(name,regs,static,{k:ops[k] for k in ('LOP3','SHF','PRMT','LDS','HMMA','HMUL2')},flush=True)
base=json.loads((D.parent/'native_audit.json').read_text())['FUSED1']['modes']['0']
out=dict(kind='Exact pre-H200 sm90a static build;not performance',cells=rows,BASE_FUSED1=base,math_decode_per16weights=dict(old_integer_SHF_LOP3=15,new_PRMT=8,new_shared_LDS=8,extra_shared_address_bias_amortized_per_fill=True,scale_integer_work_unchanged=True),bank_conflict_free_address='PRMT selects [lane*4,code,tableBaseByte2,tableBaseByte3],then add tableBaseByte1 bias;256-byte-aligned base',leaves_open='Measured net gain after64KiB table initialization,shared service anddepth-dependent residency;prior unreplicated1KiB loss is distinct')
(D/'static_audit.json').write_text(json.dumps(out,indent=2)+'\n')
