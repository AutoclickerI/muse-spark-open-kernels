"""Six sliding shapes,16 seeds, actual S11 critical-path profiles and official all-row correctness."""
import ast,hashlib,re,sys,importlib.util
from pathlib import Path
import build_all,patch_all,build_last
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/global_attention_pair.py').read_text();a=s.index('cs_CUDA=');b=s.index('# Embedded into the lab payload.',a)
d=(HERE/'all_driver.py').read_text();cell='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
cuda=build_all.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
s=s[:a]+'cs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell+'\n'+s[b:]
old_d=(HERE/'last_driver.py').read_text();old_cell='\n'.join(ast.get_source_segment(old_d,n) for n in ast.parse(old_d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):old_cell=re.sub(r'\b'+name+r'\b','ol_'+name,old_cell)
s='ol_CUDA='+repr(build_last.source())+'\nol_CPP='+repr(cpp)+'\n'+old_cell+'\n'+s

s=s.replace('CS_CASES=((1,1,True),(16,1,True),(32,1,True),(1,1,False))', 'CS_CASES=tuple((b,L,True) for L in (1,8) for b in (1,16,32))')
s=s.replace("def pp_attention(data,qkv,**kw):return attention_partial_port(qkv,data,False)", "def pp_attention(data,qkv,**kw):\n        return attention_partial_port(qkv,data,False) if data['config']['q_len']==1 else old_attention(data,qkv)")
s=s.replace("        return host.cs_attention(data,qkv,**kw) if data['config']['window'] else pp_attention(data,qkv)", "        if data['config']['q_len']==1:return ol_attention(data,qkv,**kw)\n        cfg=data['config'];plan=st['best7_plans'].get((cfg['batch'],cfg['q_len'],cfg['window']),st['best7_plans']['other'])\n        return host._best7_attention(qkv,data,st,plan)")
s=s.replace('END shapes=%d/4','END shapes=%d/6').replace('last-CTA with shared merge weights; full QT1 NS128; native WGMMA, no clusters','last-CTA sliding decode/verify; masked padded query softmax + FP32 fast sincos; native WGMMA, no clusters')
# The control state disables the host's experimental dispatch and keeps original S11 plans/weights.
s=s.replace("st=dict(st);st.pop('cluster_rows',None)","st=dict(st);st.pop('cluster_rows',None)")
compile(s,'all_sliding_layer_pair.py','exec');p=ROOT/'prof/lab/all_sliding_layer_pair.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
q=importlib.util.spec_from_file_location('_p5_full_patch',HERE.parent/'full128/patch_full128.py');pf=importlib.util.module_from_spec(q);q.loader.exec_module(pf)
host=pf.replace_cell(patch_all.patch((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()))
p=HERE/'out/s11_all_sliding_regmax.py';p.write_text(host);print('HOST',hashlib.sha256(host.encode()).hexdigest())
out=host+'\n\n'+h200_lab.WRAPPER.format(names='all_sliding_layer_pair '+tag,base='S11 + all-sliding lastCTA experimental + paired fullverify stage4',payloads=[('all_sliding_layer_pair',tag,s)],budget=285,limit=9000)
p=ROOT/'problems/muse_decode_layer/lab/lab_p5_all_sliding.py';compile(out,str(p),'exec');p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest(),len(out))
q=HERE/'screen_global.py';x=q.read_text().replace("len(ns['CS_RESULTS'])==4", "len(ns['CS_RESULTS'])==6").replace("r['case'].endswith('S')", "r['case'].startswith('d')");(HERE/'screen_all_lab.py').write_text(x)
