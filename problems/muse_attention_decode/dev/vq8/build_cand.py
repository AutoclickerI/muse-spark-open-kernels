"""Write problems/muse_attention_decode/cand_vq8.py: the scored d98ae45d submission (its prepare-time q_len-8
portfolio; one tie-sort fix) + vq8 for the shapes where vq8's import-time lab beats the incumbent dispatch."""
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
base = (ROOT / 'runs/muse_attention_decode/20261023T233000Z_submit_d98ae45d7204438d.py').read_text()
# d98ae45d's prepare() sorts (us, (splits, pack_gqa)) tuples: an exact timing tie between two pack_gqa
# settings of one split count compares None with a bool (TypeError). c9c4ef99 hit it (tests 0-5 failed).
TIE = "sorted(fa3))"
assert base.count(TIE) == 1
base = base.replace(TIE, "sorted(fa3, key=lambda x: x[0]))")
src = (HERE / 'vq8.cu').read_text()
tail = (HERE / 'cand_tail.py.inc').read_text()
out = base + '\nVQ8_CUDA = ' + repr(src) + '\n' + tail
(ROOT / 'problems/muse_attention_decode/cand_vq8.py').write_text(out)
print('wrote cand_vq8.py', len(out))
