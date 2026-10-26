import base64,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'public_ports';R=H.parents[3];sys.path.insert(0,str(R/'tools'));import h200_lab
# Rebuild the original paired-port bundle without changing its measured sources.
s=(P/'out/public_attention.py').read_text()
ns={};exec(s[:s.index('_PORT_STATE=')],ns);b=ns['BUNDLE']
b['operator_direct']=(H/'out/operator_direct.py').read_text();b['lee_hybrid']=(H/'out/lee_hybrid.py').read_text()
adapter=(P/'adapter.py.inc').read_text();adapter=adapter[:adapter.index('def covered_lee')]+(H/'policy.py.inc').read_text()
lab=(P/'lab.py.inc').read_text().replace("'compile_s':350,'run_s':460,'output_bytes':11200","'compile_s':430,'run_s':400,'output_bytes':11600").replace('8671259+7919*i','10739123+7919*i')
lab=lab.replace('global base,op,lee,pack','global base,op,lee,pack,op_direct,lee_hybrid')
lab=lab.replace(" if hook:=globals().get('PUBLIC_LOCAL_INSTALL'):hook(base,op,lee)"," op_direct=module('operator_direct',BUNDLE['operator_direct']);lee_hybrid=module('lee_hybrid',BUNDLE['lee_hybrid'])\n if hook:=globals().get('PUBLIC_LOCAL_INSTALL'):hook(base,op,lee,op_direct,lee_hybrid)")
lab=lab.replace(" op.setup('cuda');lee.setup('cuda')"," op.setup('cuda');lee.setup('cuda');op_direct.setup('cuda');lee_hybrid.setup('cuda')")
lab=lab.replace("for name in ('A','A2','O','L'):","for name in arms(d):").replace("names=('A','A2','O','L')","names=arms(d)")
a=lab.index('   # Each cohort');z=lab.index('   assert all(min(v)',a)
lab=lab[:a]+'''   # All cyclic rotations and exact reverses: every arm each exposure twice.
   for cohort in range(len(names)):
    k=(cohort+si)%len(names);order=names[k:]+names[:k]
    for seq in (order,tuple(reversed(order))):
     orders.append(list(seq))
     for name in seq:per[name].append(replay(graphs[name][0]))
'''+lab[z:]
lab=lab.replace("symmetric='4 rotated forward/reverse cohorts;8 rounds;all arms each position twice'","symmetric='N rotated forward/reverse cohorts;2N rounds;all arms each position twice'")
lab=lab.replace("('source','operator','lee','input')","('source','operator','lee','input','operator_direct','lee_hybrid')").replace('len(blob)<10800','len(blob)<11200')
lab=lab.replace('META reviewed operatorstack MIT6d9b7765 and LeeApache601c6541v56 vs exactS19; timed real producerABI;8freshseeds8symmetriccold;allroundns/TASK/AA','META reviewed licensed hybrids vs8post-port picks; directFP32partial prologue + bothKV query; producerV/earlyQ/poly25/Kmajor full;8freshseeds2N symmetriccold;TASK/allroundns/AA')
lab=lab.replace(" blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode()"," result=encode_rounds(result)\n blob=base64.b85encode(lzma.compress(json.dumps(result,separators=(',',':')).encode())).decode()")
blob=base64.b85encode(lzma.compress(json.dumps(b,separators=(',',':')).encode(),preset=9)).decode();src='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types,sys\nimport torch\nBUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(blob)+')))\n'+adapter+(H/'codec.py.inc').read_text()+lab
(H/'out/hybrid_attention.py').write_text(src)
# Static shape policy determines prospective gate counts before native timing.
ps={};exec((H/'policy.py.inc').read_text(),ps);case_arms={}
for l in (1,8):
 for batch in (1,16,32):
  for window in (2048,0):
   d=dict(config=dict(batch=batch,q_len=l,window=window,num_q_heads=16,num_kv_heads=2,head_dim=64,page_size=1));case_arms[ps['cell'](d)]=ps['arms'](d)
o=H/'out';meta=dict(payload_sha256=hashlib.sha256(src.encode()).hexdigest(),source_sha256=hashlib.sha256(b['source'].encode()).hexdigest(),seeds=[10739123+7919*i for i in range(8)],case_arms=case_arms,official_TASK=96,ABI_TASK=4*sum(map(len,case_arms.values())),scored_TASK=8*sum(map(len,case_arms.values())),identities={k:hashlib.sha256(b[k].encode()).hexdigest() for k in ('source','operator','lee','input','operator_direct','lee_hybrid')})
(o/'lab_manifest.json').write_text(json.dumps(meta,indent=2)+'\n');wrapper,_=h200_lab.build([('hybrid_attention.py',meta['payload_sha256'],src)],host='muse_bf16_proj',lab_only=True,import_budget=865);(o/'preview.py').write_text(wrapper);print(meta)
