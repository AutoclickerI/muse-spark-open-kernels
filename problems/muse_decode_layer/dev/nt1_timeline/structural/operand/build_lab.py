import base64,hashlib,json,lzma
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sha=lambda s:hashlib.sha256(s.encode()).hexdigest();cells=json.loads((D/'cells.json').read_text());counts=json.loads((D/'counts.json').read_text())
s=(R/'prof/lab/moe_nt1_reuse.py').read_text()
for v in cells.values():s+='\n'+(R/v['cell']).read_text()
review=(D.parent.parent/'review/lab.py.inc').read_text();s+='\n'+review[review.index('def review_gate'):review.index('\ndef review_nodes')]
timing=(R/'problems/muse_decode_layer/dev/s21/timing.py.inc').read_text();s+='\n'+timing[:timing.index('\ndef ab_time')]
old=(D.parent/'ladder/lab.py.inc').read_text();s+='\n'+old[old.index('def lad_hist'):old.index('\ndef lab(')]
ident=dict(S19='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a',cells={k:{a:b for a,b in v.items() if a!='cell'} for k,v in cells.items()},counts_sha256=sha((D/'counts.json').read_text()))
s+='\nOP_IDENTITIES='+repr(ident)+'\nOP_QUALIFIED='+repr(counts['qualified'])+'\n'+(D/'lab.py.inc').read_text();digest=sha(s);blob=base64.b85encode(lzma.compress(s.encode(),preset=9)).decode()
text="LAB_RESOURCES={'compile_s':110,'run_s':210,'output_bytes':11700,'context_poison':'none'}\nimport base64,lzma,linecache,hashlib\n_source=lzma.decompress(base64.b85decode("+repr(blob)+")).decode()\nassert hashlib.sha256(_source.encode()).hexdigest()=="+repr(digest)+"\n_file='<p3a-operand-"+digest[:12]+">'\nlinecache.cache[_file]=(len(_source),None,_source.splitlines(True),_file)\nexec(compile(_source,_file,'exec'),globals())\n_operand_lab=lab\ndef lab(deadline,weights):return _operand_lab(deadline,weights)\n"
p=R/'prof/lab/moe_p3a_operand.py';p.write_text(text);(D/'payload.json').write_text(json.dumps(dict(path=str(p.relative_to(R)),sha256=sha(text),expanded_sha256=digest,identities=ident,cells=cells),indent=2)+'\n');print('OPERAND_PAYLOAD',sha(text),len(text))
