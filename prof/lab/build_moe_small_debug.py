"""Build the small-M lab from the candidate's current raw-scale bank contract."""
import ast
import hashlib
from pathlib import Path

from build_moe_real_routing import namespace

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATE = ROOT / 'problems/muse_decode_layer/cand_r3_best6_small_nt2.py'
OUTPUT = ROOT / 'problems/muse_prefill_layer/lab/lab_small_m_nt4.py'


def select(source, names):
    parts = []
    found = set()
    for node in ast.parse(source).body:
        defined = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else {
            n.id for t in getattr(node, 'targets', []) for n in ast.walk(t) if isinstance(n, ast.Name)
        }
        if defined & names:
            parts.append(ast.get_source_segment(source, node))
            found.update(defined)
    assert names <= found, names - found
    return '\n\n'.join(parts) + '\n'


def build():
    source = CANDIDATE.read_text()
    names = {
        'H', 'NLOC', 'TOPK', 'TILE_BYTES', 'NT1', 'NCNT', 'V9_CUDA_SRC', 'V9_CPP_SRC',
        '_V9_MOD', '_v9_module', '_DQ_SLOTS', '_LA_MASK', '_bit_table', '_TABLE',
        '_repack', 'v9_prepare', 'v9_kernel_fn', 'V9_NT2_CUDA_SRC', 'V9_NT2_CPP_SRC',
        '_BEST5_NT2_MOD', '_best5_nt2_module', 'SMALL_NT4_CUDA_SRC', 'SMALL_NT4_CPP_SRC',
        '_SMALL_NT4_MOD', '_small_nt4_module', 'HOT_NT1_CUDA_SRC', 'HOT_NT1_CPP_SRC',
        '_HOT_NT1_MOD', '_hot_nt1_module', 'hot_nt1_kernel_fn', 'v9_small_nt1_kernel_fn',
        'v9_nt4_kernel_fn',
    }
    body = select(source, names)
    assert '_encode_scales' not in body and 'u = sf.contiguous().view(torch.uint8)' in body
    for node in ast.parse(body).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            if node.targets[0].id.endswith('CUDA_SRC'):
                assert '__byte_perm(sw' not in node.value.value, node.targets[0].id
    fixture = (ROOT / 'problems/muse_decode_layer/moe_tune.py').read_text()
    fixture = fixture[:fixture.index('\ndef _graph_time')]
    # Reuse just the public fixture and graph helpers, with no obsolete expert banks.
    old = (HERE / 'moe_real_routing_driver.py').read_text()
    helpers = select(old, {'lab_router', 'lab_datasets', 'lab_board', 'LabGraph', 'lab_pair'})
    parts = [
        '"""Small-M raw E4M3 contract and stage diagnostics; no layer state is returned."""\n',
        'import torch\nfrom torch.utils.cpp_extension import load_inline\n',
        'import time as lab_time\nimport math as lab_math\nimport statistics as lab_stats\n',
        'lab_torch = torch\nIS_HOPPER = torch.cuda.get_device_capability() == (9, 0)\n',
        '# Candidate source SHA-256 ' + hashlib.sha256(source.encode()).hexdigest() + '\n',
        body, namespace(fixture, 'fx_'), helpers,
        (HERE / 'moe_small_debug_driver.py.inc').read_text(),
    ]
    OUTPUT.write_text('\n'.join(parts))
    print(OUTPUT.relative_to(ROOT), hashlib.sha256(OUTPUT.read_bytes()).hexdigest())
    return OUTPUT


if __name__ == '__main__':
    build()
