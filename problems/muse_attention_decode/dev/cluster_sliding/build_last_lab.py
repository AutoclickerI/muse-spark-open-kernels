"""Paired S11 lab for the one-launch last-CTA reduction, without cluster residency or DSM."""
import ast,hashlib,re
from pathlib import Path
import build_last
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/compact_sliding_timeline.py').read_text();lines=s.splitlines(True)
# Replace its embedded driver and kernel, leaving original S11/CUPTI helpers unchanged.
a=s.index('cs_CUDA=');b=s.index('# Embedded into the lab payload.',a)
d=(HERE/'last_driver.py').read_text();tree=ast.parse(d)
cell='\n'.join(ast.get_source_segment(d,n) for n in tree.body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
cuda=build_last.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
s=s[:a]+'cs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell+'\n'+s[b:]
a=s.index('CS_CLOCK_CUDA=');b=s.index('def lab(deadline,weights,state=None):',a);s=s[:a]+s[b:]
s=s.replace('max_workers=4','max_workers=2').replace(';clock_fs=[pool.submit(cs_clock_module,w) for w in (False,True)]','').replace('        for f in clock_fs:f.result()\n','')
s=s.replace("            phase_rows=[cs_phase(qkv,data,w,host,st) for w in (False,True)]\n            lines.append('PH '+label+' '+json.dumps(phase_rows,separators=(',',':')))\n",'')
s=s.replace('tuple(172903+7919*i for i in range(4))','tuple(172903+7919*i for i in range(16))').replace('four seeds x8','sixteen seeds x8')
s=s.replace('compact cooperative Q/K/V projection reduction, native WGMMA + DSM, auto split; W4/W8 force 4/8 splits', 'last-CTA publish/retire/reduce, native WGMMA, no clusters; auto ~2 SM waves, W4/W8 fixed split count')
s=s.replace("                for n in ratios:drift[n].append(cs_delta(out[n],out['A']))", "                assert all(bool(torch.isfinite(x).all()) for value in out.values() for x in value[:2]), ('nonfinite replay',label,seed)\n                for n in ratios:drift[n].append(cs_delta(out[n],out['A']))")
compile(s,'last_sliding_timeline.py','exec');p=ROOT/'prof/lab/last_sliding_timeline.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
