"""Keep the frozen scored program intact; embed DFlash compute diagnostics."""
import ast
import hashlib
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
base=ROOT/'problems/muse_bf16_proj/cand_fused_select.py'
source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9'
dense=(ROOT/'serving/sglang/python/sglang/srt/layers/muse_proj.py').read_text()
draft_original=(ROOT/'serving/sglang/python/sglang/srt/layers/muse_dflash.py').read_text()
draft=draft_original
draft=draft.replace('from __future__ import annotations\n','')
draft=draft.replace('from sglang.srt.layers import muse_proj as _df_proj\n','')
source+='\nimport types as _df_types\n_df_proj=_df_types.ModuleType("embedded_muse_proj")\n'
source+='exec('+repr(dense)+',_df_proj.__dict__)\n\n'+draft
# The counterfactual pointwise/bmm bodies are exactly the official pinned stock
# source, compiled with the same default Inductor backend; no local imports.
stock=subprocess.check_output(['git','-C',str(ROOT/'serving/sglang'),'show',
                               'a55ab7d5:python/sglang/srt/models/dflash.py'],text=True)
for old,new in (('_grouped_conv','_df_stock_grouped_conv'),('_score_edges','_df_stock_score_edges')):
    node=next(n for n in ast.parse(stock).body if isinstance(n,ast.FunctionDef) and n.name==old)
    text=ast.get_source_segment(stock,node).replace('def '+old+'(', 'def '+new+'(',1)
    source+='\n@torch.compile(dynamic=True)\n'+text+'\n'
source+='\n_DF_MODULE_SHA='+repr(hashlib.sha256(draft_original.encode()).hexdigest())+'\n'
source+='\n'+(ROOT/'problems/muse_bf16_proj/dev/dflash_diag_tail.py').read_text()
out=ROOT/'problems/muse_bf16_proj/cand_dflash_diag.py'
compile(source,str(out),'exec')
with out.open('x') as f:f.write(source)
print(out.relative_to(ROOT),hashlib.sha256(source.encode()).hexdigest())
