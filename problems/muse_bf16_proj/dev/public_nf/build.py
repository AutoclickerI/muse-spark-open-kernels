"""Route B: transform our own direct-BF16 kernel; never reads external source.
Experiment removes activation staging. The lossless format, MMA ownership,
escape correction, split outputs and PDL contract remain ours.
"""
import ast
from pathlib import Path
H = Path(__file__).resolve().parent
base = (H.parent / 'coded_dense/direct_body.py').read_text().split('\n"""Lossless register-A WGMMA experiment;', 1)[0]
tree = ast.parse(base)
node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == 'CD_CUDA' for t in n.targets))
cuda = ast.literal_eval(node.value)
start = cuda.index('  const int nch = (nb + CB - 1) / CB;', cuda.index('coded_gemv('))
end = cuda.index('\n  float acc[RT][NT][4];', start)
cuda = cuda[:start] + '''  // Independent register activation path. Invalid MMA columns duplicate the
  // final valid token; only valid columns are stored. No zero row is needed.
  if (prew) pdl_wait();
  const __nv_bfloat16* xrow[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
    xrow[nt] = X + (size_t)min(g + 8 * nt, M - 1) * x_stride + k0 + 16 * t;
''' + cuda[end:]
cuda = cuda.replace('xv[nt][0] = src[0];\n          xv[nt][1] = src[1];', '''asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(xv[nt][0].x), "=r"(xv[nt][0].y), "=r"(xv[nt][0].z), "=r"(xv[nt][0].w) : "l"(src));
          asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(xv[nt][1].x), "=r"(xv[nt][1].y), "=r"(xv[nt][1].z), "=r"(xv[nt][1].w) : "l"(src + 1));''')
cuda = cuda.replace('__bfloat162float(xs[tok * RS + (k - k0)])', '__bfloat162float(X[(size_t)tok * x_stride + k])')
# V3 only. All K partials remain visible to the layer's existing finalizers.
cuda = cuda.replace('const int R = N / 16, smem = smem_bytes(M, KB, S, V, NW * RT * 16);', 'const int R = N / 16, smem = 0;')
lines = base.splitlines(keepends=True)
body = ''.join(lines[:node.lineno-1]) + 'CD_CUDA = ' + repr(cuda) + '\n' + ''.join(lines[node.end_lineno:])
body = body.replace('for v in (3, 4)', 'for v in (3,)')
(H / 'register_x.py').write_text(body)
(H / 'register_x.cu').write_text(cuda)
print('Built own direct-BF16 register-X kernel')
