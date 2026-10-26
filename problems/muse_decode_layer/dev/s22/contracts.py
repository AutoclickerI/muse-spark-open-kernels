"""Static contracts for S22 and all six S19 cell replacements before GPU work."""
import json,sys
import compose
H=compose.H;c=compose.load(H.parent/'s19/contracts.py','s22_s19_contracts');c.compose=compose.s19
c.DELTA_TABLES.update({'DA_SELECTED','DA_NORMALIZED','DA_ROLES','DA_REQUESTS','_P5B_FA_PICKS','_P5B_MERGE_SELECTION','_P5B_VQ_SELECTION','_P5B_FA_META','_P5B_FA_STORAGE','_P5B_FA_SMS','_PUBLIC_SELECTION','_HY_SELECTION'})
def run(dropped=(),disabled=()):
 source,meta=compose.build(dropped,disabled);arms=compose.arm_specs(dropped,disabled);texts={'S22':source,**{n:x['source'] for n,x in arms.items()}}
 parent=c.Src(compose.base());host=c.Src(source);old=c.Src(compose.expand(compose.P/'cand_r3_best7_s18_final.py'));hy=c.Src((H/'inputs/hybrid.py.inc').read_text());checks=[]
 def add(name,ok,**kw):checks.append(dict(check=name,ok=bool(ok),**kw))
 for name,text in texts.items():
  s=c.Src(text);checks.extend(dict(x,unit=name) for x in c.check_flags(s)+c.check_chain(s,old))
  checks.extend(c.check_plans(host,s,name)+c.check_banks(host,s,name))
  add('prepare_identity',s.segs('prepare')==parent.segs('prepare'),unit=name)
  for key in ('ATTN_PORT_CUDA_SRC','VQ8_CUDA','P3B_CUDA','GLUE_CL_CUDA','RVC_NFIDX0_CUDA','RVC_NBASE2_CUDA','DA_SELECTED','NP_CONFIG','_P5B_VQ_SELECTION','_PUBLIC_SOURCES','_P5B_PUBLIC_NOTICES'):
   add('retained_S21_literal',s.literal(key)==parent.literal(key),unit=name,key=key)
  for fn in ('_best7_run','_best7_project','_boundary_fast','_boundary_late','smallm_experts','_best7_boundary'):
   add('retained_S21_dispatch',s.segs(fn)==parent.segs(fn),unit=name,fn=fn)
  if name=='INC':add('incumbent_identity',compose.sha(text)==compose.BASE_SHA);continue
  picks=s.literal('_PUBLIC_SELECTION');selected=s.literal('_HY_SELECTION')
  add('d16S_restored',(16,1,True) not in picks and (16,1,True) not in selected,unit=name)
  for key in ('_HY_SOURCES','_HY_PUBLIC_NOTICES'):add('measured_hybrid_source',s.literal(key)==hy.literal(key),unit=name,key=key)
  add('complete_licenses',s.literal('_HY_PUBLIC_NOTICES')==parent.literal('_P5B_PUBLIC_NOTICES'),unit=name)
  add('measured_hybrid_dispatch',s.segs('_best7_attention')[-1]==hy.segs('_best7_attention')[-1],unit=name)
  add('delta_before_warm',text.index('_s19_prior_HYBRID=')<text.rindex('_s19_prior_warm='),unit=name)
  add('hybrid_setup_in_pool','_S19_BUILDS.append(_hy_build)' in text,unit=name)
  # Removing both interposers exposes the identical S21 pre-public dispatcher.
  # At all six cells it is S19 (the sole remaining VQ override is v32F).
  add('S19_dispatch_chain',s.segs('_best7_attention')[:-1]==parent.segs('_best7_attention'),unit=name)
  add('no_old_FA_overrides',s.literal('_P5B_FA_PICKS')=={} and s.literal('_P5B_MERGE_SELECTION')=={},unit=name)
  if name.startswith('off_'):
   cell=arms[name]['cell'];key=compose.CELLS[cell]
   add('cell_is_S19',key not in picks and key not in selected and key!=(32,8,False),unit=name)
   hp=host.literal('_PUBLIC_SELECTION');hh=host.literal('_HY_SELECTION');hp.pop(key);hh.pop(key,None)
   add('one_cell_only',picks==hp and selected==hh,unit=name)
  if name.startswith('no_HY_'):
   cell=arms[name]['cell'];key=compose.CELLS[cell];hh=host.literal('_HY_SELECTION');hh.pop(key)
   add('hybrid_leaveout_prior_port',selected==hh and picks==host.literal('_PUBLIC_SELECTION') and key in picks,unit=name)
 add('primary_public_selection',host.literal('_PUBLIC_SELECTION')=={k:v for k,v in parent.literal('_PUBLIC_SELECTION').items() if k!=(16,1,True) and k not in {compose.CELLS[n] for n in dropped}})
 return dict(passed=all(x['ok'] for x in checks),checks=checks,source_sha256=meta['sha256'])
if __name__=='__main__':
 d=json.loads((H/'config.json').read_text()) if (H/'config.json').exists() else {};r=run(d.get('dropped_cells',[]),d.get('disabled_hybrids',[]));(H/'contracts.json').write_text(json.dumps(r,indent=1)+'\n');bad=[x for x in r['checks'] if not x['ok']];print('PASS' if r['passed'] else 'FAIL',len(r['checks']),bad[:12]);sys.exit(not r['passed'])
