from pathlib import Path
p=Path(__file__).resolve().parent.parent
native=(p/'cand_v6.py').read_text()
dq=native[native.index('__device__ __forceinline__ void dq8'):native.index('\nstruct Route')]
# Includes only decoder; keep the canonical physical bit permutation.
(p/'dev/canonical_decoder.inc').write_text(dq)
asm=[]
for n in (8,16,32,64,128):
 c=n//2
 regs=', '.join('%'+str(i) for i in range(c))
 ar=', '.join('%'+str(c+i) for i in range(4))
 outputs=', '.join('"+f"(d['+str(i)+'])' for i in range(c))
 clause='if constexpr' if n==8 else 'else if constexpr'
 asm.append(f'''  {clause} (N == {n}) {{
    asm volatile("{{ .reg .pred p; setp.ne.b32 p, 1, 0;\\n"
      "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{c+4}, p, 1, 1, 0; }}\\n"
      : {outputs}
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc));
  }}''')
(p/'dev/grouped_atoms.inc').write_text('\n'.join(asm))
cu=(p/'dev/canonical_grouped.cu').read_text().replace('// @DECODER@',dq).replace('// @ATOMS@','\n'.join(asm))
cu=cu.replace('#include <ATen/cuda/CUDAContext.h>','#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAException.h>')
cu=cu.replace('uint32_t A[2][4];','uint32_t A[2][4] = {};')
base=native.replace('def prepare(weights):','def _prepare_native(weights):').replace('def kernel_fn(data, state):','def _native_call(data, state):')
# Give this separate module an explicit name; native weight packing is unchanged.
host=(p/'dev/grouped_host.py').read_text()
(p/'cand_rs.py').write_text(base+'\nGROUPED_SRC = r"""'+cu+'"""\n'+host)
(p/'dev/grouped_expanded.cu').write_text(cu)
