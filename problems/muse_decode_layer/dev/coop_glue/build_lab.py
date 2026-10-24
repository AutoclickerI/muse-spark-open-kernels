from pathlib import Path
import hashlib,sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
old=HERE.parent/'fused_glue';route=HERE.parent/'route_glue'
paired=(HERE.parent/'paired_lab/driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
shared=(old/'driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
shared=shared[:shared.index('\ndef fg_setup')]+shared[shared.index('\ndef fg_route_gate'):]
setup=(route/'driver.py.inc').read_text().split('\ndef lab(deadline,weights,host_state):',1)[0]
# Pin all source strings now. The queued file will not reread lane work.
body='BEST7_SOURCE='+repr((ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text())+'\n'
body+='LEGACY_SOURCE='+repr('FG_CUDA='+repr((route/'projection.cu').read_text())+'\n'+(old/'runtime.py.inc').read_text())+'\n'
body+=(route/'dropin.py').read_text().replace('return out,ids,wts,lids,meta',"state['route_last_meta']=meta\n    return out,ids,wts,lids,meta")+'\nRG_CUDA='+repr((route/'route_once.cu').read_text())+'\n'
body+=(HERE/'dropin.py').read_text()+'\n'+paired+'\n'+shared+'\n'
body+=(old/'flip_gate.py.inc').read_text()+'\n'+setup+'\n'+(HERE/'driver.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(body),rejected_words(body)
out=ROOT/'prof/lab/best7_coop_glue.py';out.write_text(body)
manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),host='muse_bf16_proj')
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
