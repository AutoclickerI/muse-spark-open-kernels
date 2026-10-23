"""Build the owned single-file V11 dense portfolio without changing lane files."""
import argparse
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def build(output):
    parent = (ROOT/'problems/muse_prefill_layer/cand_v11.py').read_text()
    source = parent
    replacements = {
        'qkv = torch.mm(x, st["qkv"].t())': 'qkv = _pg_layer_linear(x, st, "qkv_gate")',
        'a = torch.mm(o_in, st["o"].t())': 'a = _pg_layer_linear(o_in, st, "o")',
        'f = torch.mm(moe, st["unpooler"].t())': 'f = _pg_layer_linear(moe, st, "unpooler")',
    }
    for old,new in replacements.items():
        assert source.count(old)==1, old
        source=source.replace(old,new)
    a=source.index('    logits = torch.mm(h, st["router_cat"].t(), out_dtype=torch.float32)')
    z=source.index('    if T < 8192:',a)
    source=source[:a]+'''    pooled, ids, topk_w, slot_e, slot_pos, counts, _ = _pg_layer_group(h, st, data["local_start"])
'''+source[z:]
    node=next(n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef) and n.name=='_routing_kernel')
    route=ast.get_source_segment(parent,node).replace('def _routing_kernel(', 'def _pg_routing_256(',1)
    old='tl.load(logits_ptr + t * 512 + e) + tl.load(logits_ptr + t * 512 + 256 + e)'
    assert route.count(old)==1
    route=route.replace(old,'tl.load(logits_ptr + t * 256 + e)')
    module=ROOT/'serving/sglang/python/sglang/srt/layers/muse_prefill_gemm.py'
    block=module.read_text().replace('from __future__ import annotations\n','')
    (ROOT/'problems/muse_bf16_proj/dev/prefill_gemm_block.py').write_text(block)
    source+='\n\n'+block+'\n\n@triton.jit\n'+route+'\n\n'
    source+=(ROOT/'problems/muse_bf16_proj/dev/v11_gemm_tail.py').read_text()
    assert source.splitlines()[0]=='# okc: problem=muse_prefill_layer'
    compile(source,str(output),'exec')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        if output.read_text()!=source: raise FileExistsError('refusing to modify an existing candidate/fixture')
    else:
        with output.open('x') as f:f.write(source)
    print(output,hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,
        default=ROOT/'runs/screen/prefill_gemm/v11_gemm_work.py');args=ap.parse_args();build(args.output)
