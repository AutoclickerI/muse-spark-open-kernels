"""Export the serving source as a self-contained block and a local layer fixture.

The fixture lives under runs/screen, outside every lane's candidate namespace.
It is an integration test, not a judge submission or a proposed scored source.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
module = ROOT / 'serving/sglang/python/sglang/srt/layers/muse_prefill_gemm.py'
block = module.read_text().replace('from __future__ import annotations\n', '')
path = ROOT / 'problems/muse_bf16_proj/dev/prefill_gemm_block.py'
path.write_text(block)
parent = (ROOT / 'problems/muse_prefill_layer/cand_v11.py').read_text()
replacements = {
    'qkv = torch.mm(x, st["qkv"].t())': 'qkv = prefill_gemm_run(x, st["pg"], "qkv_gate")',
    'a = torch.mm(o_in, st["o"].t())': 'a = prefill_gemm_run(o_in, st["pg"], "o")',
    'logits = torch.mm(h, st["router_cat"].t(), out_dtype=torch.float32)': 'pooled, logits = prefill_gemm_run(h, st["pg"], "pool_router")',
    '    pooled = torch.mm(h, st["pooler"].t())': '    # pooled was produced together with the summed router above.',
    'f = torch.mm(moe, st["unpooler"].t())': 'f = prefill_gemm_run(moe, st["pg"], "unpooler")',
    'lg = tl.load(logits_ptr + t * 512 + e) + tl.load(logits_ptr + t * 512 + 256 + e)': 'lg = tl.load(logits_ptr + t * 256 + e)',
}
for old, new in replacements.items():
    assert parent.count(old) == 1, old
    parent = parent.replace(old, new)
source = parent + '\n\n' + block + '''

_pg_layer_parent_prepare = prepare
def prepare(weights):
    state = _pg_layer_parent_prepare(weights)
    state['pg'] = prefill_gemm_prepare(weights, tokens=(16384,3012,753), budget_s=75)
    prefill_gemm_summary(state['pg'])
    return state
'''
compile(source, 'prefill_gemm_layer_smoke.py', 'exec')
fixture = ROOT / 'runs/screen/prefill_gemm/prefill_gemm_layer_smoke.py'
fixture.parent.mkdir(parents=True, exist_ok=True)
fixture.write_text(source)
print('module', hashlib.sha256(module.read_bytes()).hexdigest())
print('block', hashlib.sha256(block.encode()).hexdigest())
print('local fixture', fixture.relative_to(ROOT), hashlib.sha256(source.encode()).hexdigest())
