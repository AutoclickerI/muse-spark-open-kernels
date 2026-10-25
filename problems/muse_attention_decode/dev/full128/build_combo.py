"""One decode check shares frozen S11's prepared state between full N128 and last-CTA sliding pairs."""
import ast,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(HERE.parent/'vq8')]
import h200_lab,partial_port
base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('acda6c62')
s=(ROOT/'prof/lab/full128_timeline.py').read_text().replace('def lab(deadline,weights,state=None):','def p5_full_pair(deadline,weights,state=None):').replace('range(8))','range(16))',1)
s=s.replace('            for g in graphs.values():g.replay()\n            torch.cuda.synchronize()', '            for g in graphs.values():g.replay()\n            torch.cuda.synchronize()\n            assert all(bool(torch.isfinite(x).all()) for value in out.values() for x in value[:2]), (\'nonfinite full replay\',label,seed)')
last=(ROOT/'prof/lab/last_sliding_timeline.py').read_text();parts=[]
for n in ast.parse(last).body:
    name=getattr(n,'name','')
    if isinstance(n,ast.Assign):
        names=[getattr(t,'id','') for t in n.targets]
        if any(k.startswith('cs_') or k.startswith('CS_') for k in names):parts.append(ast.get_source_segment(last,n))
    elif isinstance(n,ast.FunctionDef) and name in ('cs_module','cs_plan','cs_attention','cs_namespace','lab'):
        cell=ast.get_source_segment(last,n)
        if name=='lab':cell=cell.replace('def lab(','def p5_sliding_pair(',1)
        parts.append(cell)
s+='\n'+'\n\n'.join(parts)+'\n'
pp=partial_port.cell();a=pp.index('PARTIAL_FAST_CUDA = ');b=pp.index('PARTIAL_CPP = ',a);pp=pp[:a]+'PARTIAL_FAST_CUDA = PARTIAL_CUDA\n'+pp[b:]
s+='\n'+pp+'\n'+'''
def p5_pp_namespace(host):
    m=cs_namespace(host,128)
    def call(qkv,data,state,plan):
        cfg=dict(data['config'],num_q_heads=16,num_kv_heads=2)
        return attention_partial_port(qkv,dict(data,config=cfg),False)
    m._best7_attention=call
    return m
'''
s=s.replace("'W8':cs_namespace(host,128,8)}", "'W8':cs_namespace(host,128,8),'PP':p5_pp_namespace(host)}")
s=s.replace("for n in ('C64','C128','W4','W8')", "for n in ('C64','C128','W4','W8','PP')")
s+='\n'+(HERE/'combo.py.inc').read_text()
compile(s,'attention_p5_combo.py','exec');p=ROOT/'prof/lab/attention_p5_combo.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
wrapped=base+'\n\n'+h200_lab.WRAPPER.format(names='attention_p5_combo '+tag,base='frozen S11 acda6c62, shared actual state',payloads=[('attention_p5_combo',tag,s)],budget=285,limit=9000)
out=ROOT/'problems/muse_decode_layer/lab/lab_attention_p5_combo_final.py';compile(wrapped,str(out),'exec');assert len(wrapped)<1000000;out.write_text(wrapped);print(out,hashlib.sha256(wrapped.encode()).hexdigest(),len(wrapped),'payload',tag)
