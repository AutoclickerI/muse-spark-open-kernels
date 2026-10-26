"""Freeze one night batch; receipt selection changes only the parent point constants."""
import ast,base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent;R=H.parents[4];O=H/'out';O.mkdir(exist_ok=True)
def build(selection=None):
 manifest=json.loads((P/'whole_manifest.json').read_text());queue=json.loads((P/'queue.json').read_text())['sender']
 # okc_run.finish removes runs/sent after publishing the canonical receipt.
 # Native selection therefore uses that receipt's immutable source, not the
 # pre-reset snapshot used for local admission. No alternate source is accepted.
 if selection is not None and 'parent_receipt_job' in selection:
  snapshot=next((R/'runs/muse_bf16_proj').glob('*_'+selection['parent_receipt_job']+'.py'))
 else:snapshot=R/'runs/sent'/queue['sha256'][:16]/Path(queue['file']).name
 wrapper=snapshot.read_text();assert hashlib.sha256(wrapper.encode()).hexdigest()==json.loads((P/'queue.json').read_text())['sender']['sha256']
 payloads=ast.literal_eval(next(n.value for n in ast.parse(wrapper).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets)));prior=next(x[2] for x in payloads if hashlib.sha256(x[2].encode()).hexdigest()==manifest['payload_sha256']);tree=ast.parse(prior);constants={}
 for n in tree.body:
  if isinstance(n,ast.Assign):
   for t in n.targets:
    if getattr(t,'id','') in ('PP_TASK','PP_CHECK'):constants[t.id]=ast.literal_eval(n.value)
 source_node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='PP_SOURCE' for t in n.targets));packed=next(n.value for n in ast.walk(source_node) if isinstance(n,ast.Constant) and isinstance(n.value,str));base=lzma.decompress(base64.b85decode(packed)).decode();assert hashlib.sha256(base.encode()).hexdigest()==manifest['source_sha256']
 body=(H/'cluster.py.inc').read_text()+'\n'+(H/'runtime.py.inc').read_text();source=base+'\n'+body;compile(source,'<P10 night carrier kernels>','exec');(O/'source.py').write_text(source);(H/'body.py.inc').write_text(body)
 functions='\n\n'.join(ast.get_source_segment(prior,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('pp_module','pp_copy','pp_capture'))
 if selection is None:selection=dict(parent='3347492 pending: stress all15 port points',points=manifest['points']['PORTS'],pool={})
 points=selection['points'];pool={int(t):a for t,a in selection.get('pool',{}).items()};seeds=[33000319+104729*i for i in range(8)];keys=[(16,'qkv_gate'),(16,'qkv3'),(32,'qkv_gate'),(32,'qkv3'),(16,'o'),(16,'pool_router'),(32,'pool_router'),(16,'unpooler'),(32,'unpooler')]
 payload='LAB_RESOURCES={"compile_s":360,"run_s":440,"output_bytes":13744,"context_poison":"none"}\nimport torch,triton,triton.language as tl,time,types,linecache,hashlib,json,statistics,math,lzma,base64,struct,gc\nPP_SOURCE=lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(source.encode())).decode())+')).decode()\n'+''.join(k+'='+repr(v)+'\n' for k,v in constants.items())+'NG_PARENT_ID='+repr(selection['parent'])+'\nNG_PARENT_POINTS='+repr(points)+'\nNG_PARENT_POOL='+repr(pool)+'\nNG_SEEDS='+repr(seeds)+'\nNG_KEYS='+repr(keys)+'\n'+functions+'\n'+(P.parent/'native_boundary/codec.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text();compile(payload,'<P10 night whole graph batch>','exec');(O/'night_gap.py').write_text(payload)
 out=dict(payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),body_sha256=hashlib.sha256(body.encode()).hexdigest(),frozen_parent_source_sha256=manifest['source_sha256'],S21_sha256=manifest['base_sha256'],selection=selection,groups=['Q16','Q32','O16','P16','P32','U16','U32'],keys=keys,seeds=seeds,weight_seed=33000311,router_seed=33000313,protocol='all cyclic positions and exact reverse;2N rounds per case, arithmetic mean all samples',rounds={'T16':16,'T32':16,'other':8},arms=['INC:S21','A:parent portfolio','AA:parent portfolio','FULL:parent+all7new','CODED_P32:exact12 alternative/no extra bank','NO_Q16','NO_Q32','NO_O16','NO_P16','NO_P32','NO_U16','NO_U32'],parent_sender=queue['pid'],parent_lineage=3347492)
 (H/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':print(json.dumps(build(json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv)>1 else None),indent=2))
