"""patch(src) -> src: patch_vq8_verify.py's cell with the page-ring kernel (vq8.cu c262e4c5, frozen copy
out/vq8_c262e4c5.cu: page-table lookahead in a 4-byte cp.async shared ring, PD = LAG+2, no unroll) and its rows.

    python3 problems/muse_attention_decode/dev/vq8/patch_vq8_ring.py <cand.py> <out.py> [--rows 1s,1f,16f,32f]

Same ABI, prepare gate, routing and PDL contract as patch_vq8_verify.py (read its docstring); only the kernel
source and the row table differ. Evidence (proj-host lab prof/lab/vq8_layer.py, decode ABI, 4 FP32 partials,
after a 37.7 MB projection stand-in with PDL, cold L2 by read, TASK gate incl. new K/V; vs the layer's qk_prep +
public FA3 + norm_gate), lab 5928ced847984751, chain_pdl us:
  verify 1 full  (4, 2) 26.30 vs FA3 28.19   (new row; 69132eea lost here, 29.5 vs 28.5)
  verify 16 full (6, 8) 141.57 vs 191.90     (69132eea: 153.9)
  verify 32 full (6, 8) 250.62 vs 378.91     (69132eea: 278.5)
  verify 1 sliding (4, 1): its rows fell in the lab's truncated output; 69132eea measured 15.30 vs 16.83 (44aa67ba).
Not routed: verify 16 sliding 18.53 vs 16.35, verify 32 sliding 20.77 vs 21.18 (inside run-to-run spread), and
q_len 1 rows (decode 16 / 32 full 132.2 / 239.6 here vs attn_main ~104 / ~191).
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL = HERE / 'out/vq8_c262e4c5.cu'
SHA = 'c262e4c5f9c3e1f74cafa3d84689944541bf2a4c9b58fdadbe4d81122fedb269'
ROWS = {'1s': ((1, 8, True), (4, 1)), '1f': ((1, 8, False), (4, 2)), '16f': ((16, 8, False), (6, 8)),
        '32f': ((32, 8, False), (6, 8))}


def patch(src: str, rows=tuple(ROWS)) -> str:
    spec = importlib.util.spec_from_file_location('vq8_patch_cell', HERE / 'patch_vq8_verify.py')
    cell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cell)
    cu = KERNEL.read_text()
    assert hashlib.sha256(cu.encode()).hexdigest() == SHA
    cell.VQ8_CU, cell.VQ8_SHA, cell.ROWS = cu, SHA, ROWS
    return cell.patch(src, rows)


if __name__ == '__main__':
    rows = tuple(sys.argv[sys.argv.index('--rows') + 1].split(',')) if '--rows' in sys.argv else tuple(ROWS)
    out = patch(Path(sys.argv[1]).read_text(), rows)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
