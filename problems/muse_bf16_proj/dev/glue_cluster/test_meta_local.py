"""Local check: route_cl's MoERouteMeta equals nt1_branches' make_route_meta on the same local ids (the 711 field
words; the rest of the 2,944 B is alignment padding neither writes),
eager and in 6 graph replays (the arrival counter resets), t = 1..32."""
import pathlib
import torch
from torch.utils.cpp_extension import load_inline

HERE = pathlib.Path(__file__).resolve().parent
NT1 = HERE.parents[2] / 'muse_decode_layer/dev/nt1_branches'
CPP = (HERE / 'patch_glue_cl.py').read_text().split("CPP = r'''")[1].split("'''")[0]
cuda = (HERE / 'glue_cl.cu').read_text()
ref_src = ('#include <cuda_runtime.h>\n#include <stdint.h>\n' + (NT1 / 'route_meta.h').read_text() +
           (NT1 / 'route_meta.cu.inc').read_text() +
           '\nvoid ref_meta(int64_t ids, int64_t M, int64_t start, int64_t dst){make_route_meta<<<1,256>>>((const int*)ids,(int)M,(int)start,(MoERouteMeta*)dst);}\n')
ext = load_inline(name='glue_cl_meta_test', cpp_sources=CPP, cuda_sources=cuda, functions=['route_cl'],
                  extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr'], verbose=False)
ref = load_inline(name='route_meta_ref_test', cpp_sources='#include <cstdint>\nvoid ref_meta(int64_t,int64_t,int64_t,int64_t);',
                  cuda_sources=ref_src, functions=['ref_meta'], extra_cuda_cflags=['-O3'], verbose=False)
dev = 'cuda'
bad = 0
for t in (1, 2, 8, 16, 31, 32):
    for start in (0, 96, 224):
        g = torch.Generator(device=dev).manual_seed(t * 7 + start)
        S = 8
        P = torch.randn((S, t, 4608), device=dev, generator=g)
        P[:, :, 4096 + start:4096 + start + 32] += 0.8  # make local experts likely
        bias = torch.zeros(256, device=dev)
        prew = torch.ones(4096, device=dev)
        ls = torch.tensor([start], device=dev, dtype=torch.int32)
        x = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
        ids = torch.empty((t, 8), device=dev, dtype=torch.int32)
        wts = torch.empty((t, 8), device=dev)
        lids = torch.empty_like(ids)
        meta = torch.full((736,), 7777, device=dev, dtype=torch.int32)
        arrive = torch.zeros(1, device=dev, dtype=torch.int32)
        want = torch.empty(736, device=dev, dtype=torch.int32)

        def run():
            ext.route_cl(P.data_ptr(), 1, S, P.stride(0), P.stride(1), bias.data_ptr(), prew.data_ptr(), ls.data_ptr(),
                         x.data_ptr(), ids.data_ptr(), wts.data_ptr(), lids.data_ptr(), meta.data_ptr(), arrive.data_ptr(),
                         t, 8, 256, torch.cuda.current_stream().cuda_stream)
        run()
        torch.cuda.synchronize()
        ref.ref_meta(ids.data_ptr(), t, start, want.data_ptr())
        torch.cuda.synchronize()
        ok = torch.equal(meta[:711], want[:711]) and int(arrive.item()) == 0
        if not ok:
            d = (meta[:711] != want[:711]).nonzero().flatten()[:8].tolist()
            print('diff at', d, meta[d].tolist(), want[d].tolist())
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            run()
        for i in range(6):
            P.normal_(generator=g)
            P[:, :, 4096 + start:4096 + start + 32] += 0.8
            meta.fill_(-5)
            gr.replay()
            torch.cuda.synchronize()
            ref.ref_meta(ids.data_ptr(), t, start, want.data_ptr())
            torch.cuda.synchronize()
            ok &= torch.equal(meta[:711], want[:711]) and int(arrive.item()) == 0
        pairs = int(meta[3].item())
        print(f't={t} start={start} pairs={pairs} {"OK" if ok else "MISMATCH"}')
        bad += not ok
print('ALL OK' if bad == 0 else f'{bad} MISMATCH')
