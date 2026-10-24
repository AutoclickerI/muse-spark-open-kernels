"""Audit saved-board dispatch and the precise changes inherited by best7."""
from pathlib import Path
import ast, hashlib, json, subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
BOARD = ROOT / 'runs/muse_decode_layer/20261023T185641Z_submit_0e94268cf5654012.py'
BASE = ROOT / 'problems/muse_decode_layer/cand_r3_best6.py'

def native(path, name):
    return next(ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))

result = {'board_sha256': hashlib.sha256(BOARD.read_bytes()).hexdigest(), 'variants': {}}
signature = ('template __global__ void moe_fused<{nt}>(const __nv_bfloat16*, const int*, const float*, '
             'const __nv_bfloat16*, const uint8_t*, const uint8_t*, const float*, const float*, '
             '__nv_bfloat16*, float*, float*, int*, __nv_bfloat16*, int, int, int);\n')
for name, path, literal, nt in (('board_nt1', BOARD, 'V9_CUDA_SRC', 1),
                               ('best6_nt1', BASE, 'V9_CUDA_SRC', 1),
                               ('best6_nt2', BASE, 'V9_NT2_CUDA_SRC', 2)):
    source = native(path, literal)
    source = source[:source.index('size_t smem_bytes(')]
    source = source.replace('#include <torch/extension.h>', '#include <cuda_runtime.h>')
    source = source.replace('#include <ATen/cuda/CUDAContext.h>', '')
    source += 'static_assert(sizeof(Route) == 1864);\n' + signature.format(nt=nt) + '\n}\n'
    cu = HERE / (name + '_local.cu')
    cu.write_text(source)
    command = ['nvcc', '-std=c++20', '-O3', '-arch=sm_90a', '--cubin', '-Xptxas=-v', str(cu), '-o', str(cu.with_suffix('.cubin'))]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (HERE / (name + '.log')).write_text(compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    print(name, compiled.stderr.strip())
    # The actual grid is 132 * cudaOccupancyMaxActiveBlocksPerMultiprocessor.
    # NT1's launch bound caps registers at 128, permitting two 256-thread CTAs.
    smem = {m: 2*36864 + 2*8*32*nt*4*4 + 32 + 1864 + ((m*8+15)//16)*16 + 4*m*8 for m in (32,128)}
    result['variants'][name] = dict(nt=nt, token_block=8*nt, threads=256, weight_stages=2,
        weight_tile_bytes=36864, w13_output_rows=16, w2_output_rows=16,
        smem_model_bytes=smem, ptxas=compiled.stderr, source_sha256=hashlib.sha256(source.encode()).hexdigest())
(HERE / 'dispatch_audit_local.json').write_text(json.dumps(result, indent=2)+'\n')
