import base64,hashlib,json,lzma
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];cells=json.loads((D/'cells.json').read_text());s=(R/'prof/lab/moe_nt1_reuse.py').read_text()
ec=json.loads((D/'early/cells.json').read_text());cells.update({k:v for k,v in ec.items() if k.startswith('ear')})
for v in cells.values():s+='\n'+(R/v['cell']).read_text()
review=(D.parent.parent/'review/lab.py.inc').read_text();s+='\n'+review[review.index('def review_gate'):review.index('\ndef review_nodes')]
timing=(R/'problems/muse_decode_layer/dev/s21/timing.py.inc').read_text();s+='\n'+timing[:timing.index('\ndef ab_time')]
ident=dict(base_S19='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a',cells=cells);s+='\nLAD_IDENTITIES='+repr(ident)+'\n'+(D/'lab.py.inc').read_text()+'\n'+(D/'hist.py.inc').read_text();digest=hashlib.sha256(s.encode()).hexdigest();blob=base64.b85encode(lzma.compress(s.encode(),preset=9)).decode();payloads=[]
for part in (0,1):
 p=R/f'prof/lab/moe_p3a_ladder{part}.py';text=f"LAB_RESOURCES={{'compile_s':65,'run_s':140,'output_bytes':{8600 if part==0 else 2800},'context_poison':'none'}}\nLAD_PART={part}\nimport base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode({blob!r})).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=={digest!r}\n_file='<p3a-ladder-{digest[:12]}>'\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,'exec'),globals())\n_lad_lab={'lab' if part==0 else 'lab_hist'}\ndef lab(deadline,weights):return _lad_lab(deadline,weights)\n";p.write_text(text);payloads.append(dict(part=part,path=str(p.relative_to(R)),sha256=hashlib.sha256(text.encode()).hexdigest(),expanded_sha256=digest))
(D/'payloads.json').write_text(json.dumps(dict(payloads=payloads,identities=ident),indent=2)+'\n');print('PAYLOADS',[(v['part'],v['sha256'][:12]) for v in payloads])
