"""Static contracts for S23 and its leave-out arms before any GPU work."""
import ast,difflib,json,sys
import compose
H=compose.H
TOKENS={'d1':1,'d16':16,'d32':32,'v1':8,'v16':128,'v32':256}

def literal(source,name):
 return next(ast.literal_eval(n.value) for n in reversed(ast.parse(source).body) if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==name)

def changed(a,b):
 return [x for x in difflib.unified_diff(a.splitlines(),b.splitlines(),n=0,lineterm='') if x[:1] in '+-' and x[:3] not in ('+++','---')]

def run():
 checks=[]
 def add(name,ok,**kw):checks.append(dict(check=name,ok=bool(ok),**kw))
 sel=compose.selection();host,meta=compose.build();arms=compose.arm_specs()
 own=sys.modules['compose'];sys.modules['compose']=compose.s22  # s22/contracts.py imports its own compose
 try:s22c=compose.load(H.parent/'s22/contracts.py','s23_s22_contracts').run(['v16F'],['d32S'])
 finally:sys.modules['compose']=own
 add('S22_base_contracts',s22c['passed'],count=len(s22c['checks']))
 base=compose.base(True);add('host_extends_S22_hybrid_off_base',host.startswith(base+'\n'))
 add('incumbent_identity',compose.sha(arms['INC']['source'])==compose.BASE_SHA)
 add('incumbent_is_S22_phase1_shipment',compose.sha(compose.expand(compose.R/compose.BASE_FILE))==compose.BASE_SHA)
 tail=host[len(base)+1:]
 for p in compose.HELPERS:add('helper_verbatim',p.read_text() in tail,helper=str(p.relative_to(compose.R)))
 notice=compose.HELPERS[0].read_text()
 add('MIT_notice_shipped',all(x in notice for x in ('MIT License','Copyright (c) 2026 bigboateng','Permission is hereby granted')))
 add('ports_match_P10_selection',literal(host,'S23_PORTS')=={(t,r):p for t,r,p in map(tuple,sel['points'])})
 add('night_matches_P10_selection',literal(host,'S23_NIGHT')==tuple(sel['groups']))
 add('hybrid_d32S_disabled',(32,1,True) not in literal(host,'_HY_SELECTION') and (32,1,True) in literal(arms['INC']['source'],'_HY_SELECTION'))
 add('bank_delta_last',host.rindex('_s19_prior_P10PORTS=_s16_prepare_delta')>host.rindex('_s19_prior_warm=_s16_prepare_delta'))
 night=literal(compose.HELPERS[1].read_text(),'NG_GROUPS')
 for name,spec in arms.items():
  text=spec['source'];compile(text,name,'exec')
  if name=='INC':continue
  g=spec['group'];diff=changed(host,text)
  if g=='HYd32S':
   add('one_change',len(diff)==2 and all(x[1:].replace(' ','').startswith('_HY_SELECTION=') for x in diff),arm=name,diff=diff)
   keys={(32,1,True)}
   add('rows_cover_change',spec['rows']==['d32S'],arm=name)
  else:
   lit='S23_NIGHT' if g in compose.NIGHT else 'S23_PORTS'
   add('one_change',len(diff)==2 and all(x[1:].replace(' ','').startswith(lit+'=') for x in diff),arm=name,diff=diff)
   ts={t for t,r in night[g]} if g in compose.NIGHT else {compose.PORTS[g][0]}
   add('rows_cover_change',sorted(spec['rows'])==sorted(l+s for l,t in TOKENS.items() if t in ts for s in 'SF'),arm=name,tokens=sorted(ts))
 return dict(passed=all(x['ok'] for x in checks),checks=checks,source_sha256=meta['sha256'])

if __name__=='__main__':
 r=run();(H/'contracts.json').write_text(json.dumps(r,indent=1)+'\n');bad=[x for x in r['checks'] if not x['ok']];print('PASS' if r['passed'] else 'FAIL',len(r['checks']),bad[:8]);sys.exit(not r['passed'])
