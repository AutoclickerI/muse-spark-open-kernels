"""Capacity arithmetic from serving/RULES.md and docs/reference/expert_layout.md."""
import json
from pathlib import Path
kv=15872;bank_per_layer=905969664;layers=60;bank=bank_per_layer*layers;budget=141_000_000_000*.85
rows=[]
for concurrency in (1,4,16,32):
 for length in (80000,80000+753,80000+753+220,131072):
  count=concurrency*length;size=count*kv
  rows.append(dict(concurrency=concurrency,tokens_per_request=length,total_tokens=count,kv_bytes_per_gpu=size,kv_gb=size/1e9,kv_gib=size/2**30,
    remainder_one_bank_bytes=budget-bank-size,remainder_two_banks_bytes=budget-2*bank-size))
out=dict(kv_bytes_per_token_per_gpu=kv,bank_bytes_per_layer=bank_per_layer,bank_bytes_60_layers=bank,
 static_budget_bytes=budget,extra_bank_token_equivalent=bank/kv,
 increment_16k_chunk_bytes=16384*kv,increment_220_output_bytes=220*kv,
 assumptions='No prefix sharing or sliding-cache savings; context cap includes generation; hidden judge lengths and actual residency unmeasured.',scenarios=rows)
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
