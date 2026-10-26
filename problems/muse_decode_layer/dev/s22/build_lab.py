"""One S22 native batch, with S19 per-cell counterfactuals and fresh phase seeds."""
import ast,base64,difflib,json,lzma,sys
from pathlib import Path
import compose
H=compose.H;R=compose.R;P=compose.P;sha=compose.sha
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
from pack import pack
import h200_lab
PROTOCOL=dict(name='palindromic-cohort-v1',rounds=16,seed_count=64,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)

def seed_manifest(phase):
 selected=set();review=[]
 def collect(v,key=''):
  if isinstance(v,dict):
   for k,x in v.items():collect(x,k)
  elif isinstance(v,list):
   if key in ('seeds','seed_ids','confirmation_seeds','selection_seeds'):selected.update(x for x in v if isinstance(x,int))
   else:
    for x in v:
     if isinstance(x,(dict,list)):collect(x)
 for directory in ('s18','s19','s20','s21'):
  collect(json.loads((H.parent/directory/'seed_manifest.json').read_text()))
 for p in sorted((R/'problems').rglob('*.json')):
  if p.name in ('P4_HANDOFF.json','point_nominations.json'):
   collect(json.loads(p.read_text()));review.append(dict(path=str(p.relative_to(R)),sha256=sha(p.read_bytes())))
 for p in sorted((H/'stages').glob('p*/manifest.json')):
  v=json.loads(p.read_text())
  if v['phase']<phase:collect(v)
 start=203400003+100000000*phase;seeds=[start+104729*i for i in range(64)]
 assert not selected.intersection(seeds)
 (H/'handoff_review.json').write_text(json.dumps(dict(files=review,selection='S21+d16S restore+9b043487 direct-input hybrids'),indent=2)+'\n')
 return dict(selection_seeds=sorted(selected),confirmation_seeds=seeds,weight_seed=start-2,router_seed=start+2)

def main():
 cfg=json.loads((H/'config.json').read_text());phase=cfg['phase'];host=cfg['host'];budget=820.
 source,meta=compose.build(cfg['dropped_cells'],cfg['disabled_hybrids']);specs=compose.arm_specs(cfg['dropped_cells'],cfg['disabled_hybrids']);arms={}
 (H/'generated').mkdir(exist_ok=True);(H/'generated/s22.py').write_text(source)
 shipped='\n'.join(x.rstrip() for x in pack(source).splitlines())+'\n';candidate=P/'cand_r3_best7_s22_final.py';candidate.write_text(shipped)
 lines=source.splitlines(True)
 for name,spec in specs.items():
  text=spec['source'];other=text.splitlines(True)
  edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal']
  restored=list(lines)
  for a,b,t in reversed(edits):restored[a:b]=[t]
  assert ''.join(restored)==text
  arms[name]=dict(sha256=sha(text),edits=edits,**{k:v for k,v in spec.items() if k!='source'})
 problem=R/'problems/muse_decode_layer';ref='\n'.join(x for x in (problem/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
 checker='\n'.join(x for t in ((problem/'checks.py').read_text(),(problem/'nvfp4.py').read_text(),ref) for x in t.splitlines() if not x.startswith('from __future__ import'))
 sd=seed_manifest(phase);hook=next(ast.literal_eval(n.value) for n in ast.parse((H.parent/'s19/build_lab.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='HOOK')
 bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=source,arms=arms),separators=(',',':')).encode(),preset=9)).decode()
 cell_arms={cell:[n for n,a in arms.items() if a['rows']==[cell]] for cell in compose.CELLS if any(a['rows']==[cell] for a in arms.values())}
 body='import torch,types,time,statistics,math\n'+(H.parent/'s19/graph_helpers.py.inc').read_text()+'\n'+(H.parent/'s21/timing.py.inc').read_text()+'\nAB_SEEDS='+repr(tuple(sd['confirmation_seeds']))+'\n'
 body+=f"S17_WEIGHT_SEED={sd['weight_seed']}\nS17_ROUTER_SEED={sd['router_seed']}\n"
 body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(checker)+'\n'
 cases=compose.load(H.parent/'nearby/cases.py','s22_nearby_cases').cases()
 body+='S17_NEARBY_CASES='+repr(cases)+"\nS17_NEARBY_ARMS=('A','INC')\nS19_HOST_HOOK="+repr(hook)+'\nS22_CELL_ARMS='+repr(cell_arms)+'\nS22_CELLS='+repr(compose.CELLS)+'\n'
 core=(H.parent/'s21/lab.py.inc').read_text()
 old="    result['original_pair_repeat']=s21_repeat_original_case(checker,base,deadline)\n";assert core.count(old)==1;core=core.replace(old,'')
 old="    result['measurement_protocol']=dict(S21_PROTOCOL)\n";assert core.count(old)==1
 extra="""    result['cell_ratio_scale']=1000000
    result['cell_ratios']={}
    for label,names in S22_CELL_ARMS.items():
        b,q,sliding=S22_CELLS[label];key=(b,q,2048 if sliding else 0)
        if key in times:
            result['cell_ratios'][label]={name:[round(row[name]/row['A']*1000000)-1000000 for row in times[key]] for name in ['A2']+names}
"""
 core=core.replace(old,old+extra).replace("'META S21 on exact S19;64 disjoint held-out seeds;all-round symmetric cohorts;public TASK all arms;restricted leave-outs take A2 off their rows'","'META S22 vs exact S21;64 fresh seeds x16 symmetric rounds;S19 per-cell leave-outs;TASK all arms'").replace("'S21_XZ85 '+blob","'S22_XZ85 '+blob")
 body+=(H.parent/'s21/evidence.py').read_text()+'\n'+core
 encoded=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 payload='LAB_RESOURCES={"compile_s":300,"run_s":520,"output_bytes":11700,"context_poison":"none"}\nimport base64,lzma,linecache,hashlib\n_body=lzma.decompress(base64.b85decode('+repr(encoded)+')).decode()\n'
 payload+='assert hashlib.sha256(_body.encode()).hexdigest()=='+repr(sha(body))+'\n_f="<s22-lab-'+sha(body)[:12]+'>"\nlinecache.cache[_f]=(len(_body),None,_body.splitlines(True),_f)\nexec(compile(_body,_f,"exec"),globals())\n_run=lab\ndef lab(deadline,weights,state):return _run(deadline,weights,state)\n'
 pp=H/'generated/s22_pair.py';pp.write_text(payload);wrapper,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],host,phase='import',import_budget=budget,lab_only=True)
 (H/'generated/wrapper.py').write_text(wrapper);assert len(wrapper.encode())<1_000_000
 labels=['d1S','d1F','v1S','v1F','d16S','d16F','d32S','d32F','v16S','v16F','v32S','v32F']
 expected=64*sum(2+sum(not a['rows'] or label in a['rows'] for a in arms.values()) for label in labels)
 meta.update(phase=phase,host=host,import_budget=budget,shipped_source=str(candidate.relative_to(R)),shipped_sha256=sha(shipped),payload=str(pp.relative_to(R)),payload_sha256=sha(payload),wrapper='generated/wrapper.py',wrapper_sha256=sha(wrapper),sent_wrapper=f'problems/{host}/lab/lab_s22_pair.py',
  checker_sha256=sha(checker),baseline_arm='INC',nearby_expected_checks=260,nearby_arms=['A','INC'],arms={n:{k:v for k,v in a.items() if k!='edits'} for n,a in arms.items()},
  **sd,expected_TASK_checks=expected,measurement_protocol=PROTOCOL,drop_rule='Per-cell time(leave-out)/time(S22) upper CI95<1; select then confirm on new disjoint seeds',official_TASK_run=None)
 (H/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');(H/'selection.json').write_text(json.dumps({k:meta[k] for k in ('series','sha256','base_sha256','dropped_cells','disabled_hybrids')},indent=2)+'\n')
 stage=H/'stages'/f'p{phase}';stage.mkdir(parents=True,exist_ok=True);(stage/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
 print('S22',meta['sha256'],meta['shipped_sha256'][:16],meta['wrapper_sha256'][:16],len(wrapper),expected)
if __name__=='__main__':main()
