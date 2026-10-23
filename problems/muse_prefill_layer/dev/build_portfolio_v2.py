"""Pack both BR4 SwiGLU warp schedules into DG's synthetic expert selection."""
from pathlib import Path
P=Path(__file__).resolve().parents[1]
base=(P/'cand_expert_portfolio.py').read_text()
old="('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048')"
new="('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048','dg_w8','dg_hint_w8','dg_k2048_w8','dg_hint_k2048_w8')"
assert base.count(old)==2
v=base.replace(old,new)
old='D=D,BR=4,num_warps=4)'
assert v.count(old)==1
v=v.replace(old,"D=D,BR=4,num_warps=8 if '_w8' in name else 4)")
v=v.replace('BN64/128/256 and timed BF16 scratch.', 'BN64/128/256, timed BF16 scratch and packed SwiGLU occupancy choices.')
(P/'cand_expert_portfolio_v2.py').write_text(v)
full=(P/'cand_prefill_portfolio.py').read_text()
assert full.count("('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048')")==2
full=full.replace("('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048')",new)
full=full.replace(old,"D=D,BR=4,num_warps=8 if '_w8' in name else 4)")
(P/'cand_prefill_portfolio_v2.py').write_text(full)
# Test wrappers take an explicit candidate path; the production source remains frozen.
s=(P/'dev/verify_expert_portfolio.py').read_text().replace("tuple(m._GROUPED_CHOICES)+('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048')", "tuple(m._GROUPED_CHOICES)+"+new)
s=s.replace("P/'cand_expert_portfolio.py'","P/'cand_expert_portfolio_v2.py'")
(P/'dev/verify_expert_portfolio_v2.py').write_text(s)
s=(P/'dev/verify_prefill_portfolio.py').read_text().replace('cand_prefill_portfolio.py','cand_prefill_portfolio_v2.py')
(P/'dev/verify_prefill_portfolio_v2.py').write_text(s)
print('built v2, 18 same-bank expert configurations')
