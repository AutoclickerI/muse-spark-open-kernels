"""Export the standalone V9-phase + persistent grouped TMA experiment."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def wgmma():
    out = []
    for n in (8, 16, 32, 64):
        regs = n // 2
        dst = ','.join(f'%{i}' for i in range(regs))
        src = ','.join(f'%{regs+i}' for i in range(4))
        outputs = ','.join(f'"+f"(d[{i}])' for i in range(regs))
        inputs = ','.join(f'"r"(a[{i}])' for i in range(4))
        out.append(f'''if constexpr (N=={n}) {{
    asm volatile("{{ .reg .pred p; setp.ne.b32 p,%{regs+5},0; "
      "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dst}}},{{{src}}},%{regs+4},p,1,1,0; }}"
      :{outputs}:{inputs},"l"(desc),"r"(scale));
  }}''')
    return '\n  else '.join(out)


def build(output=None):
    base = (ROOT/'problems/muse_moe_w4a16/cand_v9_phase.py').read_text()
    other = (ROOT/'problems/muse_prefill_layer/cand_expert_tma_cluster.py').read_text()
    tree = ast.parse(other)
    names = {'_raw_blob_index','_raw_cta_row_order','_raw_repack','_encode_blob_fast',
             '_encode_sf_fast','repack_fast'}
    fragments = ['import triton\nimport triton.language as tl\n']
    lines = other.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            first = min([node.lineno]+[d.lineno for d in node.decorator_list])
            fragments.append(''.join(lines[first-1:node.end_lineno]))
    # Exact router-realistic fixture builder, embedded with no repository imports.
    fixture = (ROOT/'problems/muse_decode_layer/moe_tune.py').read_text()
    for node in ast.parse(fixture).body:
        if isinstance(node, ast.FunctionDef) and node.name in ('_rms','_basis','make_fixture','route_summary'):
            fragments.append(ast.get_source_segment(fixture, node))
    fragments.append('''HIDDEN=8192
D_EXPERT=4096
N_EXPERTS=256
LOCAL_EXPERTS=32
TOP_K=8
ROUTER_MULTIPLIER=0.1767766952966369
RMS_EPS=1e-5
STREAM_SEED=5354
STREAM_MEAN=0.7
STREAM_SHARED=0.9
STREAM_RANK=4
DRAFT_SHARE=0.6
''')
    cuda = (HERE/'grouped.cu').read_text().replace('// @GENERATED_WGMMA@', wgmma())
    fragments.append('TG_CUDA = '+repr(cuda))
    fragments.append((HERE/'tail.py').read_text())
    path = Path(output) if output else ROOT/'problems/muse_moe_w4a16/cand_tma_grouped.py'
    source = base+'\n\n'+'\n\n'.join(fragments)+'\n'
    if path.exists() and path.read_text()!=source:
        raise SystemExit('Refusing to overwrite a frozen candidate; use --output with a new name')
    if not path.exists():path.write_text(source)
    return path


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output')
    print(build(parser.parse_args().output))
