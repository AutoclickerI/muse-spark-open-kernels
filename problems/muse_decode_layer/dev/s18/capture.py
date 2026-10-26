"""Pin lane nominations, patch sources, and exact parents before composing."""
import ast,hashlib,json,sys,subprocess,tempfile
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path.insert(0,str(H.parent/'s16'))
import capture as old
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()

def main():
 s15=(P/'cand_r3_best7_s15.py').read_text();s14=(P/'cand_r3_best7_s14_final.py').read_text()
 def from_patch(p):
  with tempfile.TemporaryDirectory(dir=H/'generated') as t:
   root=Path(t);f=root/'problems/muse_decode_layer/cand_r3_best7_s15.py';f.parent.mkdir(parents=True);f.write_text(s15)
   z=subprocess.run(['patch','--silent','-p1','-d',str(root),'-i',str(p)],text=True,capture_output=True);assert z.returncode==0,(p,z.stderr,z.stdout)
   return f.read_text()
 paths={'C':H.parent/'pdl_release/confirmation/C.patch','Vwait':R/'problems/muse_attention_decode/dev/full_attention/rebuild/ready_elide.patch'}
 cells={};prov={}
 for name,p in paths.items():
  new=from_patch(p);cells[name]=old.delta(s15,new);prov[name]=dict(patch=str(p.relative_to(R)),patch_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_sha256=sha(new));print(name,[(x['name'],x['old'] is None) for x in cells[name]])
 for name,p in [('pdl_M',H.parent/'gate_audit/generated/nominee_M.py'),('pdl_D',H.parent/'gate_audit/generated/nominee_D.py')]:
  new=p.read_text();cells[name]=old.delta(s14,new);prov[name]=dict(source_sha256=sha(new));print(name,[(x['name'],x['old'] is None) for x in cells[name]])
 manifest=json.load(open(H.parent/'nt1_timeline/review/next_issue/manifest.json'))
 for name,key,i in [('small_index','INDEX',0),('small_shard','BASE',0),('small_pair','BASE',1),('small_block','BASE',2)]:
  m=manifest[key]['variants'][i];t=(R/m['cell']).read_text();assert sha(t)==m['cell_sha256'];cells[name]=t;prov[name]=m
 p=R/'problems/muse_attention_decode/dev/sliding_round2/out/s14_p5_decode_points.py';t=p.read_text();assert sha(t)=='c2000b3c4d64c292ccac0a61295df60d8478b232ed7e9fdf0de118806fa924f4';prov['P5']=dict(source_sha256=sha(t),receipt='27a914f72bfb47c8')
 (H/'cells.json').write_text(json.dumps(cells,separators=(',',':'))+'\n');(H/'cells_manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256((H/'cells.json').read_bytes()).hexdigest(),provenance=prov),indent=2)+'\n')
if __name__=='__main__':main()
