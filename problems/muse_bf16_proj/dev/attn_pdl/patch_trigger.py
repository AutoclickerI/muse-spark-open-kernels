"""Apply the attention early-release patch to a decode-layer candidate.

    python3 problems/muse_bf16_proj/dev/attn_pdl/patch_trigger.py <cand.py> <out.py>

attn_main (ATTN_PORT_CUDA_SRC) executes `griddepcontrol.launch_dependents` at entry, so a
successor launched with programmatic stream serialization (S6's M=1 O cell, GF config6) can become
resident during attention and load its whole weight slice before its own `griddepcontrol.wait`.
Every PDL successor in the candidate waits before reading attention's output (GF: `griddepcontrol.wait`
before the activation loads; Triton kernels: gdc_wait), and non-PDL successors (cuBLAS) keep full
stream order, so the patch changes only scheduling, never values.
"""
import hashlib
import sys
from pathlib import Path

OLD = 'attn_main(const Params p) {\n'
NEW = ('attn_main(const Params p) {\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n'
       '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n')


def patch(src: str) -> str:
    start = src.index('ATTN_PORT_CUDA_SRC = r"""')
    end = src.index('"""', start + 30)
    body = src[start:end]
    if body.count(OLD) != 1:
        raise SystemExit(f'expected one attn_main entry, found {body.count(OLD)}')
    if 'griddepcontrol.launch_dependents' in body:
        raise SystemExit('attention already releases its dependents')
    return src[:start] + body.replace(OLD, NEW) + src[end:]


if __name__ == '__main__':
    src = Path(sys.argv[1]).read_text()
    out = patch(src)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
