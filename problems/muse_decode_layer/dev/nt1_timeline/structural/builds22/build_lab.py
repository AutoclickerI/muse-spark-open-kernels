"""One whole-production-graph S22 cohort for every screened conditional P3a build."""
import ast,base64,difflib,hashlib,importlib.util,json,lzma,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_builds22';sys.path.insert(0,str(R/'tools'));import h200_lab
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
PROTOCOL=dict(name='palindromic-cohort-v1',rounds=16,seed_count=64,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)

def checker():
 p=R/'problems/muse_decode_layer';ref='\n'.join(x for x in (p/'reference.py').read_text().splitlines() if not x.startswith(('from checks import','from nvfp4 import')))
 return '\n'.join(x for t in ((p/'checks.py').read_text(),(p/'nvfp4.py').read_text(),ref) for x in t.splitlines() if not x.startswith('from __future__ import'))

def main():
 candidates=json.loads((D/'manifest.json').read_text());active=json.loads((D/'active.json').read_text()) if (D/'active.json').exists() else list(candidates);assert active and set(active)<=set(candidates)
 base=(O/'s22.py').read_text();assert all(v['base_sha256']==sha(base) for v in candidates.values());lines=base.splitlines(True);arms={}
 for name in active:
  v=candidates[name];text=(R/v['candidate']).read_text();assert sha(text)==v['candidate_sha256'];other=text.splitlines(True)
  edits=[(a,b,''.join(other[c:d])) for op,a,b,c,d in difflib.SequenceMatcher(None,lines,other,autojunk=False).get_opcodes() if op!='equal'];restored=list(lines)
  for a,b,t in reversed(edits):restored[a:b]=[t]
  assert ''.join(restored)==text;arms[name]=dict(sha256=sha(text),edits=edits,rows=[])
 bundle=base64.b85encode(lzma.compress(json.dumps(dict(host=base,arms=arms),separators=(',',':')).encode(),preset=9)).decode();check=checker()
 seeds=[267310003+104729*i for i in range(64)];selected=set(range(8))|{257310003,257414732,1779003,1783739}
 for directory in ('s18','s19','s20','s21','s22'):
  path=R/'problems/muse_decode_layer/dev'/directory/'seed_manifest.json'
  if not path.exists():continue
  v=json.loads(path.read_text());selected.update(v.get('selection_seeds',[]));selected.update(v.get('confirmation_seeds',[]))
 assert not selected.intersection(seeds)
 s21=R/'problems/muse_decode_layer/dev/s21';s19=s21.parent/'s19';hook=next(ast.literal_eval(n.value) for n in ast.parse((s19/'build_lab.py').read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='HOOK')
 cp=s21.parent/'nearby/cases.py';spec=importlib.util.spec_from_file_location('p3a22_nearby',cp);cases=importlib.util.module_from_spec(spec);spec.loader.exec_module(cases);nearby=cases.cases()
 timing=(s21/'timing.py.inc').read_text().replace("('A','A2','INC')","('A','A2',TRACE_CANDIDATE)").replace("primary['INC']","primary[TRACE_CANDIDATE]")
 body='import torch,types,time,statistics,math\n'+(s19/'graph_helpers.py.inc').read_text()+'\n'+timing+'\nTRACE_CANDIDATE='+repr(active[0])+'\nAB_SEEDS='+repr(tuple(seeds))+'\nS17_WEIGHT_SEED=267310001\nS17_ROUTER_SEED=267310005\n'
 body+='import base64,lzma,json\nS16_BUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(bundle)+')))\nS16_CHECKER='+repr(check)+'\nS19_HOST_HOOK='+repr(hook)+'\n'
 body+='S17_NEARBY_CASES='+repr(nearby)+'\nS17_NEARBY_ARMS='+repr(tuple(['A']+active))+'\n'
 counts=json.loads((D/'counts.json').read_text())['rows'];basecounts=json.loads((D.parent/'operand/counts.json').read_text())['rows'];resources=counts+[dict(v,arm='A') for v in basecounts if v['arm']=='BASE'];body+='P3A_EXPECTED='+repr([{k:v[k] for k in ('M','arm','registers','stack','shared','static','resident')} for v in resources])+'\n'
 core=(s21/'lab.py.inc').read_text();core=core.replace("    result['original_pair_repeat']=s21_repeat_original_case(checker,base,deadline)\n",'')
 core=core.replace('S16_LOCAL_INSTALL(m)',"globals()['S16_LOCAL_INSTALL'](m)").replace('S21_LOCAL_ARCHIVE(result)',"globals()['S21_LOCAL_ARCHIVE'](result)")
 anchor="    for name,spec in S16_BUNDLE['arms'].items():";assert core.count(anchor)==1;core=core.replace(anchor,"    p3a_delta={}\n"+anchor+"\n        p3a_start=time.monotonic()")
 anchor='            states[name]=st';assert core.count(anchor)==1;core=core.replace(anchor,anchor+'\n            p3a_delta[name]=time.monotonic()-p3a_start')
 anchor='    generator=base[\'_ap_generate_input\']';assert core.count(anchor)==1
 probe='''    p3a_resources=[]
    for name,v in ns.items():
        for M,prefix,F in ((1,'rvc_nfidx0',95),(8,'rvc_nbase1',69),(16,'rvc_nbase2',69),(32,'rvc_nbase2',69)):
            info=list(v[prefix+'_module']().epi_info(F,2,M))
            if torch.cuda.get_device_capability(device)==(9,0):
                ex=next(x for x in P3A_EXPECTED if x['arm']==name and x['M']==M)
                assert info==[ex['resident'],ex['registers'],ex['stack'],ex['shared'],ex['static']],(name,M,info,ex)
            p3a_resources.append([name,M,info])
'''
 core=core.replace(anchor,probe+anchor)
 anchor="    result['measurement_protocol']=dict(S21_PROTOCOL)\n";assert core.count(anchor)==1
 core=core.replace(anchor,anchor+"    result['p3a_resources']=p3a_resources\n    result['p3a_prepare_delta_s']=p3a_delta\n    result['p3a_prepare_upper_s']={n:host_prepare_s+t for n,t in p3a_delta.items()}\n")
 anchor="    names=list(next(iter(loads.values()))['samples'][0]) if loads else [];labels=list(loads)";assert core.count(anchor)==1
 moments='''    result['p3a_shape_moments_nano']={}
    for key,per in times.items():
        if key[0]*key[1]>32:continue  # no P3a code change here;all six-load samples still retained
        result['p3a_shape_moments_nano'][ab_label(key)]={}
        for name in ['A2']+list(arms):
            logs=[math.log(row[name]/row['A']) for row in per]
            result['p3a_shape_moments_nano'][ab_label(key)][name]=[round(statistics.mean(logs)*1e9),round(statistics.stdev(logs)/len(logs)**.5*1e9)]
'''
 core=core.replace(anchor,moments+anchor)
 # Only discard recomputable summaries, after complete checks have evaluated them.
 anchor="    names=list(next(iter(loads.values()))['samples'][0]) if loads else [];labels=list(loads)"
 assert core.count(anchor)==1
 core=core.replace(anchor,"    for shape in result['shapes'].values():shape.pop('r',None);shape.pop('aa',None)\n    for key in ('prepare_lines','phase_s','build_s','delta_s','attention_plan','shared_jit','p3a_prepare_upper_s'):result.pop(key,None)\n"+anchor)
 core=core.replace("'S21_XZ85 '+blob","'P3A22_XZ85 '+blob").replace("'META S21 on exact S19;64 disjoint held-out seeds;all-round symmetric cohorts;public TASK all arms;restricted leave-outs take A2 off their rows'","'META P3a whole S22 production graphs;64 fresh seeds x16 symmetric rounds;all12shapes allarms;TASK/nearby controls;no per-call score transfer'")
 core=core.replace('s21_pack_evidence(compact(result))','s21_pack_evidence(p3a_pack_vectors(compact(result)))')
 body+=(s21/'evidence.py').read_text()+'\n'+(D/'sample_codec.py').read_text()+'\n'+core
 digest=sha(body);blob=base64.b85encode(lzma.compress(body.encode(),preset=9)).decode()
 payload="LAB_RESOURCES={'compile_s':350,'run_s':490,'output_bytes':11744,'context_poison':'none'}\nimport base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode("+repr(blob)+")).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=="+repr(digest)+"\n_f='<p3a-s22-"+digest[:12]+">'\nlinecache.cache[_f]=(len(_source),None,_source.splitlines(True),_f)\nexec(compile(_source,_f,'exec'),globals())\n_p3a_lab=lab\ndef lab(deadline,weights,state):return _p3a_lab(deadline,weights,state)\n"
 pp=R/'prof/lab/moe_p3a_s22.py';pp.write_text(payload);wrapper,_=h200_lab.build([(pp.stem,sha(payload)[:16],payload)],'muse_moe_w4a16',phase='import',import_budget=850.,lab_only=True);wp=O/'wrapper.py';wp.write_text(wrapper);assert len(wrapper.encode())<1_000_000
 meta=dict(series='P3A22',sha256=sha(base),base_sha256=sha(base),baseline_arm='A2',arms={n:dict(sha256=a['sha256'],rows=[]) for n,a in arms.items()},active=active,host='muse_moe_w4a16',payload=str(pp.relative_to(R)),payload_sha256=sha(payload),expanded_payload_sha256=digest,wrapper=str(wp.relative_to(R)),wrapper_sha256=sha(wrapper),checker_sha256=sha(check),selection_seeds=sorted(selected),confirmation_seeds=seeds,weight_seed=267310001,router_seed=267310005,expected_TASK_checks=64*12*(2+len(arms)),nearby_arms=['A']+active,nearby_expected_checks=len(nearby)*2*(1+len(arms)),measurement_protocol=PROTOCOL,comparison='Host A/A2 are exact S22. Arm time/S22 ratios invert to speedup;all12 whole graphs measured for every arm.',import_budget=850.,candidate_sources={n:candidates[n] for n in active})
 (D/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print('WHOLE_GRAPH',active,meta['payload_sha256'][:16],meta['wrapper_sha256'][:16],len(wrapper),meta['expected_TASK_checks'],meta['nearby_expected_checks'],flush=True)
if __name__=='__main__':main()
