"""Compile the frozen submitted producers for Hopper; inspect SASS and spills."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]


def main():
    tree = ast.parse((ROOT / 'prof/lab/moe_m128_scale_bank.py').read_text())
    code = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'SCALE_CUDA' for t in n.targets))
    path = HERE / 'native.cu'; path.write_text(code)
    p = subprocess.run(['nvcc', '--cubin', '-O3', '-std=c++20', '-arch=sm_90a', '-lineinfo', '-Xptxas=-v',
                        str(path), '-o', str(path.with_suffix('.cubin'))], capture_output=True, text=True)
    path.with_suffix('.ptxas.log').write_text(p.stdout + p.stderr); p.check_returncode()
    assert 'C7513' not in p.stderr and not re.search(r'[1-9]\d* bytes spill', p.stderr), p.stderr
    sass = subprocess.run(['cuobjdump', '--dump-sass', str(path.with_suffix('.cubin'))],
                          capture_output=True, text=True, check=True).stdout
    path.with_suffix('.sass').write_text(sass)
    regs = {}; name = None
    for line in p.stderr.splitlines():
        if 'Function properties for ' in line: name = line.split('Function properties for ', 1)[1].strip()
        match = re.search(r'Used (\d+) registers', line)
        if match: regs[name] = int(match[1])
    rows = []
    for part in sass.split('Function : ')[1:]:
        name = part.splitlines()[0].strip()
        if 'fused' not in name: continue
        ops = re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;', part)
        row = dict(name=name, registers=regs[name], local_memory=sum(op.startswith(('LDL','STL')) for op,arg in ops),
                   mma=sum(op.startswith('HGMMA.') for op,arg in ops), shared_loads=sum(op.startswith('LDS') for op,arg in ops))
        assert row['local_memory'] == 0 and row['mma'] > 0, row
        rows.append(row)
    assert len(rows) == 12, len(rows)
    import sys
    sys.path.insert(0, str(HERE.parent))
    from check_async import sass_paths
    paths = [p for kernel in sass_paths(path.with_suffix('.sass')) for p in kernel['paths']]
    assert len(paths) == 20 and all(len(p['banks']) == 2 and p['hazard_count'] == 0 for p in paths)
    (HERE / 'native.json').write_text(json.dumps(dict(sha256=hashlib.sha256(code.encode()).hexdigest(), kernels=rows,
        K64_paths=len(paths), register_lifetime_hazards=0), indent=2) + '\n')
    print('SCALE_NATIVE_PASS', len(rows), 'Hopper specializations; no spills or C7513', flush=True)


if __name__ == '__main__':
    main()
