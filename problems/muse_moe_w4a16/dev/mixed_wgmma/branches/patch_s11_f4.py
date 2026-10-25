"""Apply H200 5e197b5e's F4 finalizer to S11 final's nf cell.

Usage: python3 patch_s11_f4.py INPUT OUTPUT
Only nf_plan's finalizer split changes. NF_CUDA already contains the measured
finish<4> implementation; every channel keeps its arithmetic and reduction order.
"""
import ast
import hashlib
import sys
from pathlib import Path


def patch(text):
    tree = ast.parse(text)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nf_plan')
    lines = text.splitlines(keepends=True)
    cell = ''.join(lines[node.lineno - 1:node.end_lineno])
    old = 'mod.br_finish(fp,m,start,1,torch.cuda.current_stream().cuda_stream)'
    assert cell.count(old) == 1, 'expected S11 nf_plan finalizer'
    new = cell.replace(old, 'mod.br_finish(fp,m,start,4,torch.cuda.current_stream().cuda_stream)')
    cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'NF_CUDA' for t in n.targets))
    assert 'BR_FIN(4)' in cuda and 'template<int SPLIT>__global__ void finish' in cuda
    lines[node.lineno - 1:node.end_lineno] = [new]
    result = ''.join(lines)
    compile(result, '<S11 F4>', 'exec')
    return result


if __name__ == '__main__':
    src, dst = map(Path, sys.argv[1:])
    dst.write_text(patch(src.read_text()))
    print(dst, hashlib.sha256(dst.read_bytes()).hexdigest())
