"""Add genuinely new fused expert algorithms, keeping the queued portfolio immutable."""
from pathlib import Path
import ast
p=Path(__file__).resolve().parents[1]
s=(p/'cand_score_portfolio.py').read_text()
anchor="    st['sms']=torch.cuda.get_device_properties(st['qkv'].device).multi_processor_count\n"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'    _prepare_fused_banks(st,weights)\n    _prepare_overlap(st)\n')
s=s.replace("st['expert_dispatch']={753:'raw11',3012:'raw11',16384:'raw_v11'}", "st['expert_dispatch']={753:'fused_bf16_64',3012:'fused_bf16_64',16384:'fused_bf16_64'}")
s+='\n'+(p/'dev/fused_expert_paths.py').read_text()+'\n'+(p/'dev/overlap_expert_paths.py').read_text()
assert s.count('flush.sum()')==1
s=s.replace('flush.sum()', '_fused_flush_l2(flush)')
ast.parse(s)
(p/'cand_fused_expert_portfolio.py').write_text(s)
print('Built new portfolio with 45 expert choices; queued source unchanged')
