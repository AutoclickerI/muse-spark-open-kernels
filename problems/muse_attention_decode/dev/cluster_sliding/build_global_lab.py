"""Canonical-S11 critical-path pair: all decode sliding loads and decode-1 full in one batch."""
import ast,base64,hashlib,lzma,re,sys
from pathlib import Path
import build_global
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=(ROOT/'prof/lab/last_sliding_timeline.py').read_text();a=s.index('cs_CUDA=');b=s.index('# Embedded into the lab payload.',a)
d=(HERE/'global_driver.py').read_text();cell='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ','_COUNTERS') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','_COUNTERS','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
cuda=build_global.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
s=s[:a]+'cs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell+'\n'+s[b:]
s=s.replace('CS_CASES=tuple((b,1) for b in (1,16,32))','CS_CASES=((1,1,True),(16,1,True),(32,1,True),(1,1,False))')
s=s.replace('def cs_namespace(host,nk,cs=None):','def cs_namespace(host,nk,cs=None,attn=None):\n    attn=attn or cs_attention')
s=s.replace("        if not data['config']['window']:return host._best7_attention(qkv,data,state,plan)\n",'')
s=s.replace('return cs_attention(dict(data,config=cfg),qkv,nk=nk,cs=cs).view','return attn(dict(data,config=cfg),qkv,nk=nk,cs=cs).view')
a=s.index("    variants={'A':host,");b=s.index('    CS_RESULTS=[];',a);s=s[:a]+s[b:]
s=s.replace('    for B,L in CS_CASES:', '''    st=dict(st);st.pop('cluster_rows',None)  # canonical S11 control; official host uses last-CTA only at B1 sliding
    def pp_attention(data,qkv,**kw):return attention_partial_port(qkv,data,False)
    def old_attention(data,qkv,**kw):
        return host.cs_attention(data,qkv,**kw) if data['config']['window'] else pp_attention(data,qkv)
    for B,L,sliding in CS_CASES:
        variants={'A':host,'A2':host,'C64':cs_namespace(host,64),'C128':cs_namespace(host,128),
            'W4':cs_namespace(host,128,4 if sliding else 32),'W8':cs_namespace(host,128,8 if sliding else 64),
            'OLD':cs_namespace(host,128,attn=old_attention),'PP':cs_namespace(host,128,attn=pp_attention)}''')
s=s.replace("+str(B)+'S'", "+str(B)+('S' if sliding else 'F')").replace("layer='sliding',q_len=L", "layer='sliding' if sliding else 'full',q_len=L")
s=s.replace("get((B,L,2048)","get((B,L,2048 if sliding else 0)").replace('qt=4 if B*host.HKV<=4 else 1','qt=4 if sliding and B*host.HKV<=4 else 1')
s=s.replace("('C64','C128','W4','W8')", "('C64','C128','W4','W8','OLD','PP')").replace("END shapes=%d/3", "END shapes=%d/4")
s=s.replace('single-launch sliding vs original','single-launch full/sliding vs original').replace('last-CTA publish/retire/reduce, native WGMMA, no clusters','last-CTA with shared merge weights; full QT1 NS128; native WGMMA, no clusters')
s=s.replace('TASK attention gate on real projected QKV. A2=control noise.','OLD=last sliding / direct-partials full; PP=direct-partials; TASK attention gate on real projected QKV. A2=control noise.')
sys.path.insert(0,str(HERE.parent/'vq8'));import partial_port
blob=base64.b85encode(lzma.compress(partial_port.cell().encode())).decode()
s='exec(compile(lzma.decompress(base64.b85decode('+repr(blob)+')).decode(),"<P5_frozen_partial_port>","exec"),globals())\n'+s
s='import base64,lzma\n'+s
s+='''
class _P5LiveS11:
    def __init__(self,ns):self.ns=ns
    @property
    def __dict__(self):return self.ns
    def __getattr__(self,key):return self.ns[key]
_p5_global_pair=lab
def lab(deadline,weights,state):
    global TL_EXISTING_HOST
    partial_port_ext(False)
    TL_EXISTING_HOST=(_P5LiveS11(kernel_fn.__globals__),state)
    return _p5_global_pair(deadline,weights,state)
'''
compile(s,'global_attention_pair.py','exec');p=ROOT/'prof/lab/global_attention_pair.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
sys.path.insert(0,str(HERE.parent/'full128'));import patch_full128
host=patch_full128.replace_cell((HERE/'out/s11_last_decode1.py').read_text(),(4,4))
(HERE.parent/'full128/out/s11_last_regmax.py').write_text(host)
print('HOST',hashlib.sha256(host.encode()).hexdigest())
out=host+'\n\n'+h200_lab.WRAPPER.format(names='global_attention_pair '+tag,base='S11 + paired last-CTA sliding and stage4 register-fold full verify',payloads=[('global_attention_pair',tag,s)],budget=285,limit=9000)
p=ROOT/'problems/muse_decode_layer/lab/lab_p5_global_attention.py';compile(out,str(p),'exec');p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest(),len(out));print('CUDA',hashlib.sha256(cuda.encode()).hexdigest())
