"""Shared source assembly helpers for the current cluster/light MoE builders."""
import ast
import hashlib
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
P = H.parent
R = H.parents[4]
sys.path[:0] = [str(P), str(R / 'tools'), str(H)]
from build_prefill_lab import subset  # noqa: E402

HH_CELL = P / 'm128/phase_mix/hh_cell.py'
HH_SHA = '365b623992367a1d45d3dae06a2096ddd0e6879deffab2e3924678a47997b621'
IMPORTS = ('import functools\nimport hashlib\nimport statistics\nimport time\nimport torch\nimport triton\n'
           'import triton.language as tl\nfrom torch.utils.cpp_extension import load_inline\n')


def strip_imports(code):
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Try))]
    return ast.unparse(ast.fix_missing_locations(tree))


def helpers():
    return subset((R / 'prof/lab/moe_w2fix_m256.py').read_text(), ('dc_weights', 'dc_fixtures', 'dc_oracles', 'MWGraph'))
