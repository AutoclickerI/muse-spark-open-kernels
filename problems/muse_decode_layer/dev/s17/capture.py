"""Pin the exact handoff bodies used by this S17 experiment."""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];P=H.parents[1]
sys.path.insert(0,str(H.parent/'s16'))
import capture as s16_capture
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()

def main():
    s15=(P/'cand_r3_best7_s15.py').read_text();old_near=(P/'cand_r3_best7_s15n.py').read_text()
    guarded=(H.parent/'nearby/native_validation/guarded_s15n.py').read_text();large=(H.parent/'pdl_release/confirmation/large.py').read_text()
    p5=(R/'problems/muse_attention_decode/dev/sliding_round2/out/s14_p5_verify1_dsm.py').read_text();s14=(P/'cand_r3_best7_s14_final.py').read_text()
    p7=(R/'problems/muse_bf16_proj/dev/moe_tail/runtime_generated.py.inc').read_text();ilp=(H.parent/'nt1_timeline/pilp_cell.py').read_text()
    assert sha(guarded)=='9cc90d2e4764f0b47016ec526aa9742a86394ec87c3e838bf66c6376a672bfc8'
    assert sha(p5)=='403bb778b295b87adabf668f338fe423fc70e91c72007f9a6ca331c4ff5f1baf'
    assert sha(p7)=='0d5b0faf4c92b55691938bac09dfdad8ab3c5ccb312213b3012f073572ee7538'
    cells={};provenance={}
    for name,old,new in [('guarded_nearby',old_near,guarded),('pure_large_release',s15,large),('P5_35seed_v1_DSM',s14,p5)]:
        cells[name]=s16_capture.delta(old,new);provenance[name]=dict(name=name,old_sha256=sha(old),new_sha256=sha(new),nodes=[c['name'] for c in cells[name]])
    cells.update(p7_runtime=p7,ilp_cell=ilp)
    target=H/'cells.json';target.write_text(json.dumps(cells,separators=(',',':'))+'\n')
    (H/'cells_manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),provenance=provenance,p7_runtime_sha256=sha(p7),ilp_cell_sha256=sha(ilp)),indent=2)+'\n')
    print('Pinned S17 handoffs',len(cells),target.stat().st_size)
if __name__=='__main__':main()
