"""Bundle the incumbent V9 and public real-router fixture into a lab payload."""
import ast
import io
from pathlib import Path
import tokenize
import argparse

ROOT = Path(__file__).resolve().parents[4]
DECODE = ROOT / 'problems/muse_decode_layer'


def prefix(source, tag):
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return tokenize.untokenize([
        t._replace(string=tag+t.string) if t.type == tokenize.NAME and t.string in names else t
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tail', default='lab_tail.py')
    ap.add_argument('--name', default='bf16_real_grouped')
    args = ap.parse_args()
    base = (DECODE/'cand_r3_v9.py').read_text()
    v9 = base[base.index('H = 4096'):base.index('# Layer adapter: V9')]
    fixture = (DECODE/'moe_tune.py').read_text()
    fixture = fixture[fixture.index('HIDDEN = 8192'):fixture.index('def _graph_time')]
    ref = (DECODE/'reference.py').read_text()
    generator = ref[ref.index('def generate_weights'):ref.index('# Request lengths by spread')]
    generator = generator.replace('generate_weights', '_lab_make_weights').replace('_router(', '_lab_router(').replace('_stream_basis', '_lab_stream_basis').replace('random_weights(', '_lab_random_weights(')
    # The source generators are fixtures; no input/output reference is embedded.
    nv = (DECODE/'nvfp4.py').read_text()
    rnd = nv[nv.index('def random_weights'):nv.index('def outlier_channels')]
    rnd = rnd.replace('def random_weights', 'def _lab_random_weights')
    defaults = 'ROUTER_STD, ROUTER_ALIGN, ROUTER_BIAS_STD = .045, .07, .055\nSCALE_LOG2_MEAN = 7.0\nSCALE_LOG2_STD = .4\nE4M3_MAX = 448.0\nBLOCK = 16\nQ_SIZE = 1024\nQKV_SIZE = 2304\n'
    head = '"""BF16 block-product GEMM ceiling with the loaded-distribution router fixture."""\nimport torch\nimport math\nimport statistics\nimport time\nimport functools\nfrom torch.utils.cpp_extension import load_inline\n\n'
    body = prefix(defaults+fixture+generator+rnd, '_bf_')
    parts = Path(__file__).parent
    tail = (parts/args.tail).read_text()
    if args.tail in ('lab_hybrid_tail.py','lab_resources_tail.py'):
        # The candidate and the lab share one engine and one FP32 admission
        # implementation. Only the standalone payload duplicates their text.
        dense = (parts/'lab_deepgemm_tail.py').read_text()
        dense = dense.replace('for dtype in (torch.bfloat16,torch.float32):',
                              'for dtype in (torch.bfloat16,):')
        dense = dense.replace('def lab(deadline, weights):',
                              'def _bf_dense_lab(deadline, weights):')
        admission = (parts/'prepare_tail.py').read_text().split('_bf_base_prepare=prepare')[0]
        tail = dense + '\n\n' + (parts/'hybrid.py').read_text() + '\n\n' + admission + '\n' + tail
    payload = head + prefix(v9, '_bf_v9_') + '\n' + body + '\n' + tail
    out = ROOT/'prof/lab'/f'{args.name}.py'
    out.write_text(payload)
    print(out, len(payload))


if __name__ == '__main__':
    main()
