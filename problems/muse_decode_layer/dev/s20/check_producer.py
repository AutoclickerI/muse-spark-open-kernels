"""Exercise the actual S20 evidence loop and isolate an optional namespace failure."""
import base64,hashlib,json,lzma,statistics,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from promotion import evaluate
from lab_packet import encode_packet
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
checker='''
import types
def generate_weights(*a,**k):return {'qkv_gate':types.SimpleNamespace(device='cuda')}
def generate_input(**kw):return {}
def ref_front(*a,**kw):return None
def ref_back(*a,**kw):return None
def collect(d,o):return o
def check_implementation(*a):return True,''
'''
labels={(1,1):'decode 1',(16,1):'decode 16',(32,1):'decode 32',(1,8):'verify 1',(16,8):'verify 16',(32,8):'verify 32'}
seeds=list(range(100,116));base=dict(_S16_LAB_BASE_STATE={},_candidate_prepare=lambda w:{},_ap_generate_input=lambda **kw:{})
base['kernel_fn']=lambda d,s:object()
inc=dict(base,_VQ8_ROWS={},_PP_ROWS=())
for n in ('_vq8_module','nf_module','cd_module','smallm_module','_glue_cl_ext','partial_port_ext'):inc[n]=lambda *a:None
arms={n:dict(sha256=sha(n),source=n,rows=[]) for n in ('INC','bad')}
cuda=types.SimpleNamespace(get_device_capability=lambda d:(9,0),synchronize=lambda:None,empty_cache=lambda:None)
ns=dict(torch=types.SimpleNamespace(cuda=cuda,bfloat16='bf16'),time=time,statistics=statistics,
        S16_CHECKER=checker,S16_BUNDLE=dict(host='host',arms=arms),S17_WEIGHT_SEED=1,S17_ROUTER_SEED=2,
        S17_NEARBY_CASES=[dict(label=str(i)) for i in range(65)],S17_NEARBY_ARMS=('A','INC'),AB_SEEDS=seeds,
        AB_LOADS=labels,AB_CASES=[(b,q,w) for b,q in labels for w in (2048,0)],
        ab_label=lambda c:('d' if c[1]==1 else 'v')+str(c[0])+('S' if c[2] else 'F'),
        ab_copy=lambda *a:None,ab_gate=lambda *a:'',ab_time=lambda g,r:{n:[1.02 if n=='INC' else 1.] for n in g})
class Graph:
    def replay(self):pass
ns['ab_graph']=lambda call:(Graph(),call())
exec(compile((H/'lab.py.inc').read_text(),'<S20 actual producer>','exec'),ns)
ns['s19_host']=lambda:base;ns['s16_source']=lambda spec:spec['source']
def namespace(name,source,base):
    if name=='bad':raise RuntimeError('injected optional namespace failure')
    return inc,0
ns['s16_namespace']=namespace
lines=ns['lab'](time.monotonic()+20,None,None)
raw=json.loads(lzma.decompress(base64.b85decode(lines[1].split(' ',1)[1])))
import struct
names=raw['sample_arms'];binary=base64.b85decode(raw.pop('sample_i32_b85'));values=iter(struct.unpack('<'+'i'*(len(binary)//4),binary))
for label in raw['sample_loads']:
    raw['loads'][label]['samples']=[{n:1+next(values)/raw['sample_scale'] for n in names} for _ in seeds]
manifest=dict(sha256=sha('host'),base_sha256=sha('INC'),arms=arms,checker_sha256=sha(checker),confirmation_seeds=seeds,selection_seeds=[],expected_TASK_checks=16*48,nearby_expected_checks=260)
out=evaluate(dict(id='local-producer',result=dict(passed=False)),raw,manifest)
assert not raw['complete'] and not raw['arm_status']['bad']['complete']
assert out['score']['measured'] and out['score']['CI95'][0]>1 and abs(out['score']['R']-1.02)<1e-8
assert not out['freeze_gate'] and not out['pieces']['bad']['measured']
wire=len(encode_packet('s20_pair','test',dict(lines=lines)).encode())+1
result=dict(passed=True,optional_failure_isolated=True,official_pass_required=True,score=out['score']['R'],wire_bytes=wire)
(H/'producer_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
