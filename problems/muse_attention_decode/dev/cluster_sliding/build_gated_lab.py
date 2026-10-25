"""Measure fused gate versus split gate and early versus post-preparation PDL release inside S11."""
import ast,hashlib
from pathlib import Path
import build_gated
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/compact_sliding_timeline.py').read_text();lines=s.splitlines(True)
for n in reversed(ast.parse(s).body):
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('cs_CUDA','CS_CLOCK_CUDA'):
        key=n.targets[0].id;lines[n.lineno-1:n.end_lineno]=['cs_CUDA='+repr(build_gated.source())+'\n'] if key=='cs_CUDA' else []
s=''.join(lines)
a=s.index('_CS_CLOCK_MOD={}');b=s.index('def lab(deadline,weights,state=None):',a)
helpers='''_CS_LATE_MOD={}
def cs_late_module():
    major,minor=torch.cuda.get_device_capability();arch='90a' if major==9 else f'{major}{minor}'
    if arch not in _CS_LATE_MOD:
        tag=hashlib.sha256(CS_LATE_CUDA.encode()).hexdigest()[:12]
        _CS_LATE_MOD[arch]=load_inline(name=f'cluster_sliding_{tag}_{arch}',cpp_sources=cs_CPP,cuda_sources=CS_LATE_CUDA,
            functions=['sliding_forward','sliding_resources'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}','-lineinfo'])
    return _CS_LATE_MOD[arch]

def cs_policy(host,nk,nosplit=False,late=False):
    m=cs_namespace(host,nk)
    if nosplit:m._gate_split_on=lambda data,state,plan:False if data['config']['window'] else host._gate_split_on(data,state,plan)
    if late:
        old=m._best7_attention
        def call(*args,**kw):
            previous=globals()['cs_module'];globals()['cs_module']=cs_late_module
            try:return old(*args,**kw)
            finally:globals()['cs_module']=previous
        m._best7_attention=call
    return m
'''
s=s[:a]+'CS_LATE_CUDA='+repr(build_gated.source(True))+'\n'+helpers+'\n'+s[b:]
s=s.replace('max_workers=4','max_workers=3').replace('clock_fs=[pool.submit(cs_clock_module,w) for w in (False,True)]','late_future=pool.submit(cs_late_module)').replace('for f in clock_fs:f.result()','late_future.result()')
a=s.index("    variants={'A':host");b=s.index('\n    CS_RESULTS=[]',a)
s=s[:a]+"    variants={'A':host,'A2':host,'C128':cs_policy(host,128),'F64':cs_policy(host,64,True),'F128':cs_policy(host,128,True),'L128':cs_policy(host,128,True,True),'D128':cs_policy(host,128,False,True)}"+s[b:]
s=s.replace("            phase_rows=[cs_phase(qkv,data,w,host,st) for w in (False,True)]\n            lines.append('PH '+label+' '+json.dumps(phase_rows,separators=(',',':')))\n",'')
s=s.replace("for n in ('C64','C128','W4','W8')", "for n in ('C128','F64','F128','L128','D128')")
s=s.replace('compact cooperative Q/K/V projection reduction, native WGMMA + DSM, auto split; W4/W8 force 4/8 splits', 'C128=gate split; F64/F128=gate in same attention launch; L128=fused gate + PDL release after prep; D128=split gate + late release')
s=s.replace('tuple(172903+7919*i for i in range(4))','tuple(172903+7919*i for i in range(8))').replace('four seeds x8','eight seeds x8')
s=s.replace("                for n in ratios:drift[n].append(cs_delta(out[n],out['A']))", "                assert all(bool(torch.isfinite(x).all()) for value in out.values() for x in value[:2]), ('nonfinite replay',label,seed)\n                for n in ratios:drift[n].append(cs_delta(out[n],out['A']))")
compile(s,'gated_sliding_timeline.py','exec');p=ROOT/'prof/lab/gated_sliding_timeline.py';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest(),len(s))
