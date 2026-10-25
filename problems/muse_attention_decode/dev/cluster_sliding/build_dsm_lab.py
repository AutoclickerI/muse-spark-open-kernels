"""Paired original-S11 whole-layer critical-path sweep for verify DSM gang sizes."""
import ast,hashlib,re,sys,argparse,json
from pathlib import Path
import patch_dsm,build
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('--clusters',default='{"1":16,"16":4,"32":2}',help='public batch-to-cluster map, selected after the native sweep; {} keeps only qualified host changes')
args=ap.parse_args();clusters={int(k):v for k,v in json.loads(args.clusters).items()}
assert all(b in (1,16,32) and isinstance(c,int) and c in (1,2,4,8,16) for b,c in clusters.items())
s=(ROOT/'prof/lab/last_sliding_timeline.py').read_text();a=s.index('cs_CUDA=');b=s.index('# Embedded into the lab payload.',a)
d=(HERE/'driver.py').read_text();cell='\n'.join(ast.get_source_segment(d,n) for n in ast.parse(d).body if isinstance(n,ast.FunctionDef) or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('_MOD','_PLANS','_FREQ') for t in n.targets)))
for name in ('CUDA','CPP','_MOD','_PLANS','_FREQ','module','plan','attention'):cell=re.sub(r'\b'+name+r'\b','cs_'+name,cell)
cuda=build.source();cpp='#include <cstdint>\n#include <vector>\n'+'\n'.join(re.search(r'(?:void|std::vector<int64_t>) '+fn+r'\([^)]*\)',cuda,re.S).group(0)+';' for fn in ('sliding_forward','sliding_resources'))
s=s[:a]+'cs_CUDA='+repr(cuda)+'\ncs_CPP='+repr(cpp)+'\n'+cell+'\n'+s[b:]
s=s.replace('CS_CASES=tuple((b,1) for b in (1,16,32))','CS_CASES=tuple((b,8) for b in (1,16,32))')
a=s.index("    variants={'A':host,");b=s.index('    CS_RESULTS=[];',a)
s=s[:a]+"    st=dict(st);st.pop('cluster_rows',None);st.pop('dsm_rows',None)\n    variants={'A':host,'A2':host,**{'C'+str(c):cs_namespace(host,128,c) for c in (1,2,4,8,16)}}\n"+s[b:]
s=s.replace("for n in ('C64','C128','W4','W8')", "for n in ('C1','C2','C4','C8','C16')")
s=s.replace('last-CTA publish/retire/reduce, native WGMMA, no clusters; auto ~2 SM waves, W4/W8 fixed split count','verify DSM distributed epilogue, N128 fixed clusters1/2/4/8/16, one-wave targets16/4/2')
s+='''
class _P5LiveS11:
    def __init__(self,ns):self.ns=ns
    @property
    def __dict__(self):return self.ns
    def __getattr__(self,key):return self.ns[key]
_p5_dsm_pair=lab
def lab(deadline,weights,state):
    global TL_EXISTING_HOST
    TL_EXISTING_HOST=(_P5LiveS11(kernel_fn.__globals__),state)
    return _p5_dsm_pair(deadline,weights,state)
'''
compile(s,'verify_dsm_layer_pair.py','exec');p=ROOT/'prof/lab/verify_dsm_layer_pair.py';p.write_text(s);tag=hashlib.sha256(s.encode()).hexdigest()[:16]
sys.path.insert(0,str(ROOT/'tools'));import h200_lab
host=patch_dsm.patch((ROOT/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text(),clusters);p=HERE/'out/s11_dsm_verify_regmax.py';p.write_text(host);print('HOST',hashlib.sha256(host.encode()).hexdigest())
out=host+'\n\n'+h200_lab.WRAPPER.format(names='verify_dsm_layer_pair '+tag,base='S11 + qualified B1 sliding / N128 fullverify + experimental verify DSM fixed gang',payloads=[('verify_dsm_layer_pair',tag,s)],budget=285,limit=9000)
p=ROOT/'problems/muse_decode_layer/lab/lab_p5_verify_dsm.py';compile(out,str(p),'exec');p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest(),len(out))
