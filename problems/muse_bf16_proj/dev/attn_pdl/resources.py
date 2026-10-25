"""ptxas resource usage (sm_90a and sm_120) of the lab's GF, SRG and attention sources."""
import os, re, runpy, subprocess, sys, tempfile
ns = runpy.run_path(sys.argv[1])
srcs = {'gf': ns['AP_GF_CUDA'], 'srg': ns['AP_SRG_CUDA'],
        'attn': '#include <torch/extension.h>\n' + ns['AP_ATTN_PORT_CUDA_SRC']}
d = tempfile.mkdtemp()
for arch in ('sm_90a',):
    for k, s in srcs.items():
        p = os.path.join(d, k + '.cu'); open(p, 'w').write(s)
        inc = []
        if k == 'attn':
            import torch.utils.cpp_extension as ce
            inc = sum((['-I', i] for i in ce.include_paths()), []) + ['-I', __import__('sysconfig').get_path('include')]
        r = subprocess.run(['nvcc', '-O3', '-std=c++20', '--expt-relaxed-constexpr', f'-arch={arch}', '-cubin',
                            '-Xptxas', '-v', '-o', os.devnull, p] + inc, capture_output=True, text=True)
        for line in (r.stderr or '').splitlines():
            if 'registers' in line or 'Compiling entry' in line or 'error' in line.lower():
                print(arch, k, line.strip()[:160])
