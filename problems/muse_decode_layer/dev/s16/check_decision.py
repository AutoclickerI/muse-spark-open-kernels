"""Synthetic receipt-schema conformance, never H200 performance evidence."""
import base64,json,lzma,struct
from pathlib import Path
import decide,patches
H=Path(__file__).resolve().parent

def blob(tag,value):return tag+' '+base64.b85encode(lzma.compress(json.dumps(value).encode())).decode()
def samples(n,scale=1):return [100*scale*(1+(i%3-1)*.0001) for i in range(n)]
def main():
 aa=[1+(.0001 if i%2 else -.0001) for i in range(16)]
 assert decide.paired([.9]*16,aa,16)['wins']
 assert not decide.paired([1.]*16,aa,16)['wins']
 try:decide.paired([.9]*15,aa,16)
 except AssertionError:pass
 else:raise AssertionError('incomplete seeds admitted')
 global_rows=[]
 for label in ('d1S','d16S','d32S','d1F'):
  t={k:samples(16,.9 if k=='C64' else 1) for k in ('A','A2','C64','C128','W4','W8','OLD','PP')}
  global_rows.append(dict(case=label,seeds=16,times=t,gate={},task_passed={k:16 for k in t}))
 early=dict(seeds=16,times=dict(A=samples(16),A2=samples(16)),ratios={'OLD4':{k:[.8 if k=='E4' else 1.]*16 for k in ('E4','E2','E1','L4')}})
 dsm=[dict(B=b,seeds=8,gates=[{}]*8,ratios={'A2':[1.]*8,**{'C'+str(c):[.8 if c==4 else 1.]*8 for c in (1,2,4,8,16)}}) for b in (1,16,32)]
 soft='\n'.join('R B'+str(b)+' L8 '+' '.join(k+'=100/r'+('.70000[.69000,.71000]' if k=='mix25' else '1.00000[.99990,1.00010]') for k in ('A2','mix25','mix50','gap','mixgap','tc25','ping','ping25')) for b in (16,32))
 epi=[]
 for m in (1,8,16,32):
  epi += [f'AA M{m} abs=0.010 us']
  epi += [f'PAIR M{m} {kind} us=80.000 r={.8 if kind=="NORM" else 1.:.5f} se=0.00010' for kind in ('VEC','PROD','NORM','PAIR','ONCE','ONCE_VEC','ONCE_PAIR')]
 epi+=['END seeds=16 seconds=1']
 heavy=['AA signed/abs/max=0.000/0.010/0.020','END seeds=16 used=1']
 for m in (128,256):
  heavy += [f'M{m} S14_NF n=48 us=100.00 pair=1.0000 lo/hi=1.000/1.000 L2/row=0/0 TASK=PASS bit=48']
  for kind in ('DIRECT_CLUSTER1','DIRECT_NORMAL','DIRECT_SERIAL','DEEP_SERIAL'):
   heavy += [f'M{m} {kind} n=48 us=80.00 pair={.8 if kind=="DIRECT_NORMAL" else 1.:.4f} lo/hi=0.790/{.81 if kind=="DIRECT_NORMAL" else 1.01:.3f} L2/row=0/0 TASK=PASS bit=48']
 # Exact compact binary schema used by the boundary reset batch.
 binary=[];confirm=[];bound=[]
 for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
  for w in (True,False):
   label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F')
   names=['A','A2','B','M','D'];offset=len(binary)
   for i in range(16):binary += [100000,0,0,0,0]
   confirm.append(dict(case=label,gates={},sample_columns=names,sample_offset=offset,sample_rows=16))
   if q==1 and b in (16,32):
    names=['A','A2','R','P','RP','C','S'];offset=len(binary)
    for i in range(16):binary += [100000,0,0,0,0,0,0]
    bound.append(dict(case=label,gates={},sample_columns=names,sample_offset=offset,sample_rows=16))
 data=dict(confirmation=dict(cells=confirm),boundaries=bound,binary_count=len(binary));head=json.dumps(data).encode()
 raw=struct.pack('<I',len(head))+head+struct.pack('<'+'I'*len(binary),*binary)
 bd='BD_BATCH_XZ85 '+base64.b85encode(lzma.compress(raw)).decode()
 receipts={name:dict(log=log) for name,log in {
  'global':blob('CG_XZ85',global_rows),'early':blob('EH_XZ85',early),'dsm':blob('DS_XZ85',dsm),'soft':soft,
  'epi':'\n'.join(epi),'heavy':'\n'.join(heavy),'boundary':bd}.items()}
 selected,evidence=decide.nominate(receipts)
 names={s['name'] for s in selected}
 assert names=={'global_C64','early_E4','dsm_4','soft_mix25','epi_NORM','heavy_DIRECT_NORMAL'},names
 for spec in selected:
  if spec['family'] in ('epi','reuse','heavy'):assert {r[2] for r in spec['rows']}=={True,False},spec
 # A source-wide policy keeps its true affected scope even when the stage
 # batch only measured M16/M32. S16 then has to win its M1/M8 loads too.
 policy_binary=list(binary)
 for cell in bound:
  for i in range(16):policy_binary[cell['sample_offset']+i*len(cell['sample_columns'])+2]=19999
 policy_raw=struct.pack('<I',len(head))+head+struct.pack('<'+'I'*len(policy_binary),*policy_binary)
 policy_receipts=dict(receipts,boundary=dict(log='BD_BATCH_XZ85 '+base64.b85encode(lzma.compress(policy_raw)).decode()))
 policies,_=decide.nominate(policy_receipts)
 policy=next(s for s in policies if s['name']=='boundary_R')
 assert {tuple(r) for r in policy['rows']}==set(patches.registry()['boundary_R']['rows'])
 source,meta=patches.compose(selected);compile(source,'<synthetic-selection>','exec')
 report=dict(passed=True,synthetic_only=True,performance_claim=False,selected=selected,
             checks=['selection by measured shape ratio; confirmation deferred','neutral rejected','incomplete seeds rejected','all seven receipt schemas decoded','both unchanged layer variants retained for token policies','source-wide policy retains every affected row','selected source composition parses'])
 (H/'decision_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS synthetic receipt schemas, policy scope and compound emission; not H200 evidence')
if __name__=='__main__':main()
