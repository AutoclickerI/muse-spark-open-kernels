import collections,hashlib,json,re,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_builds22';cells=json.loads((D/'cells.json').read_text());rows=[]
occ=O/'occupancy';subprocess.run(['g++','-O2','-I/usr/local/cuda/include',str(D.parent/'operand/occupancy.cpp'),'-o',str(occ)],check=True)
for key,c in cells.items():
 i=int(key[-1]);p=O/(c['arm'].lower()+str(i)+'.cu');assert hashlib.sha256(p.read_bytes()).hexdigest()==c['cuda_sha256']
 z=subprocess.run(['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o')),'-Xptxas=-v'],capture_output=True,text=True);log=z.stdout+z.stderr;p.with_suffix('.compile.log').write_text(log);z.check_returncode();sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
 f=95 if i==0 else 69;fn,body=next((n,b) for n,b in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function : |\Z)',sass,re.S) if f'moe_varILi1ELi{f}E' in n)
 v=re.search(r'Function properties for '+re.escape(fn)+r'\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*:\s*Used (\d+) registers([^\n]*)',log);assert v
 stack,ss,sl,regs=map(int,v.groups()[:4]);m=re.search(r'(\d+) bytes smem',v[5]);stat=int(m[1]) if m else 0;ops=dict(collections.Counter(x.split('.')[0] for x in re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)))
 for M in c['M']:
  stages=1 if c['arm']=='CACHE1' else 2;dyn=stages*36864+8192+2*stages*8+1904+((M*8+15)&~15)+4*M*8+(65536+128 if c['arm']=='CACHE1' else 0)
  resident=int(subprocess.check_output([str(occ),str(regs),str(dyn),str(stat)],text=True));row=dict(M=M,arm=c['arm'],registers=regs,stack=stack,spill_stores=ss,spill_loads=sl,shared=dyn,static=stat,resident=resident,ops=ops,cell_sha256=c['cell_sha256'],cuda_sha256=c['cuda_sha256'])
  # Physical register destinations are descriptive SASS,not logical chain count.
  if c['arm'] in ('LAZY','LAZY2'):row['HMMA_destination_counts']=dict(collections.Counter(re.findall(r'\bHMMA\.[^\s]+\s+(R\d+)',body)))
  rows.append(row)
 print(key,regs,stack,ss,sl,'HMMA',ops.get('HMMA',0),'FFMA',ops.get('FFMA',0),flush=True)
qualified=sorted({v['arm'] for v in rows if all(z['resident']==2 and z['spill_stores']==0 and z['spill_loads']==0 for z in rows if z['arm']==v['arm'])})
(D/'counts.json').write_text(json.dumps(dict(rows=rows,qualified=qualified,dropped=sorted({v['arm'] for v in rows}-set(qualified))),indent=2)+'\n');print('QUALIFIED',qualified,flush=True)
