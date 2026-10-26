"""Fresh paired S15/PAIR/best-independent/composition follow-up, one packed lab."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[4];sys.path.insert(0,str(D));import build
c=json.loads((D/'combo/manifest.json').read_text());m=json.loads((D/'manifest.json').read_text());text=(R/'prof/lab/moe_nt1_reuse.py').read_text()
_,cell,_=build.original();cell=cell.replace('smallm_','rv0_').replace('_SMALLM_MODULE','_RV0_MODULE').replace('SMALLM_FLAGS_BUILT','RV0_FLAGS_BUILT').replace('SMALLM_VAR_','RV0_');text+='\n'+cell
used={'PAIR'}|{s['best_independent'] for s in c['selection'].values() if s['best_independent']!='S15'}
for arm in sorted(used):
 spec=m[arm];cell=(R/spec['cell']).read_text();assert hashlib.sha256(cell.encode()).hexdigest()==spec['cell_sha256'];text+='\n'+cell
for spec in c['variants'].values():
 cell=(R/spec['cell']).read_text();assert hashlib.sha256(cell.encode()).hexdigest()==spec['cell_sha256'];text+='\n'+cell
selector=(D/'combo/selector.py.inc').read_text().replace('_combo_original_bind=smallm_bind','_combo_original_bind=rv0_bind').replace('_combo_original_experts=smallm_experts','_combo_original_experts=rv0_experts')
text+='\n'+selector
text+='\ndef rvc_best_module():\n'+''.join(f'    {m[arm]["prefix"]}_module()\n' for arm in sorted(used-{'PAIR'}))+'    pass\n'
text+='def rvc_best_bind(state,flags_by_m):\n'+''.join(f'    {m[arm]["prefix"]}_bind(state,flags_by_m)\n' for arm in sorted(used))+'    return state\n'
text+="def rvc_best_experts(data,state,flags,stages=2,grid=264):\n    m=data['hidden_states'].shape[0]\n"
for mm,spec in c['selection'].items():
 tag='rv0' if spec['best_independent']=='S15' else m[spec['best_independent']]['prefix']
 text+=f'    if m=={mm}:return {tag}_experts(data,state,flags,stages,grid)\n'
text+='    return rv0_experts(data,state,flags,stages,grid)\n'
lab=(D/'lab.py.inc').read_text();a=lab.index('REVIEW_ARMS=');b=lab.index('\n',a)
lab=lab[:a]+"REVIEW_ARMS={'PAIR':'rvp','BEST':'rvc_best','COMBO':'rvc_combo'}"+lab[b:]
lab=lab.replace('175353+7919*seed','275353+7919*seed').replace('179000+m+97*seed','279000+m+97*seed')
lab=lab.replace("record=dict(base=", "record=dict(weights_seed='275353+7919*seed',route_seed='279000+M+97*seed',base=")
lab=lab.replace('   if seed<4:',"   if 'BEST' in graphs and 'COMBO' in graphs:\n    old,new,ratio=timeline_timing(graphs['BEST'],graphs['COMBO']);rows.append([seed,m,arm_names.index('COMBO'),arm_names.index('BEST'),round(old*1000),round(new*1000),round(ratio*1e6)])\n   if seed<4:")
start=lab.index(" return ['META P3a")
lab=lab[:start]+''' return ['META P3a fresh-seed S15/PAIR/BEST/COMBO; shapes select measured point winners; COMBO itself unmeasured before this run.',
 'META BEST is the measured fastest independent S15 arm per M; COMBO combines eligible BLOCK/CANCEL/PACK5 with best PAIR/shard parent per shape. TASK only; A:A descriptive; P4 held-out six-load R confirms.',
 f'GATE seeds={done}/16 failure_count={len(failures)} max_total={max(v[0] for v in gates.values()):.7f} max_row={max(v[1] for v in gates.values()):.7f}',
 'P3A_REVIEW_XZ85 '+packed,
 f'END seeds={done} seconds={time.monotonic()-begin:.1f};nomination only,no six-load promotion']
'''
codec=(D/'receipt.py').read_text().replace("weights_seed='175353+7919*seed',route_seed='179000+M+97*seed'","weights_seed='275353+7919*seed',route_seed='279000+M+97*seed'")
text+='\n'+codec+'\n'+lab;ast.parse(text);digest=hashlib.sha256(text.encode()).hexdigest();expanded=len(text);blob=base64.b85encode(lzma.compress(text.encode(),preset=9)).decode()
text='import base64,lzma,linecache,hashlib\n_p3a_body=lzma.decompress(base64.b85decode('+repr(blob)+')).decode()\n'
text+='assert hashlib.sha256(_p3a_body.encode()).hexdigest()=='+repr(digest)+'\n_p3a_file="<p3a-combo-"+'+repr(digest[:12])+'+">"\n'
text+='linecache.cache[_p3a_file]=(len(_p3a_body),None,_p3a_body.splitlines(True),_p3a_file)\nexec(compile(_p3a_body,_p3a_file,"exec"),globals())\n_p3a_lab=lab\ndef lab(deadline,weights):\n    return _p3a_lab(deadline,weights)\n'
path=R/'prof/lab/moe_p3a_combo.py';path.write_text(text)
(D/'combo/payload.json').write_text(json.dumps(dict(payload=str(path.relative_to(R)),sha256=hashlib.sha256(text.encode()).hexdigest(),expanded_sha256=digest,expanded_bytes=expanded,packed_bytes=len(text),composition=c,arms=['PAIR','BEST','COMBO'],seeds=16,weights_seed='275353+7919*seed',route_seed='279000+M+97*seed'),indent=2)+'\n')
print(path,len(text),digest)
