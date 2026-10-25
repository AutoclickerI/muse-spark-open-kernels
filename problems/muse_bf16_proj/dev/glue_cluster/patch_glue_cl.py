"""Add the cluster glue (glue_cl.cu) to a decode-layer candidate and select it per scored row.

    python3 problems/muse_bf16_proj/dev/glue_cluster/patch_glue_cl.py <cand.py> <out.py>

A glue geometry entry ('cl', C, NT, rounds) runs boundary_cl / route_cl (one cluster of C CTAs per
token, norm partials through DSM) instead of S6's Triton `_boundary_fast` / `_route_fast` at
(warps, replicas). ROWS replaces the glue column of BEST7_SETTINGS for the listed rows; the native
module compiles in prepare's native pool beside the other modules.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (batch, q_len, window) -> ('best6', b1, route, b2); entries: (warps, replicas) = S6 Triton, ('cl', C, NT, r).
# H200 lab 1365a290 (S6 producer -> glue -> PDL consumer, paired medians vs S6's BEST7 geometry): at t <= 32
# boundaries ('cl', 8, 256, 1) -0.5 to -1.7 us each, route ('cl', 8, 256, 0) -0.3 to -1.4 us; t = 128/256 lose.
_CL = ('best6', ('cl', 8, 256, 1), ('cl', 8, 256, 0), ('cl', 8, 256, 1))
ROWS = {key: _CL for key in ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0),
                             (32, 1, 2048), (32, 1, 0))}

CPP = r'''
#include <cstdint>
void boundary_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t NB, int64_t stream, int64_t sout,
                 int64_t out, int64_t postw, int64_t alpha, int64_t beta, int64_t prew, int64_t T, int64_t C, int64_t NT,
                 int64_t rounds, int64_t stream_handle);
void route_cl(int64_t P, int64_t pf32, int64_t S, int64_t ps, int64_t rs, int64_t bias, int64_t prew, int64_t lstart,
              int64_t X, int64_t ids, int64_t wts, int64_t lids, int64_t meta, int64_t arrive, int64_t T, int64_t C,
              int64_t NT, int64_t stream_handle);
'''

BLOCK = '''

# ---- P7 cluster glue (problems/muse_bf16_proj/dev/glue_cluster): one cluster of C CTAs per token ----
GLUE_CL_ROWS = __ROWS__
for _key, _glue in GLUE_CL_ROWS.items():
    BEST7_SETTINGS[_key] = (_glue,) + tuple(BEST7_SETTINGS[_key][1:])
GLUE_CL_CUDA = __CUDA__
GLUE_CL_CPP = __CPP__
_GLUE_CL_EXT = None


def _glue_cl_ext():
    global _GLUE_CL_EXT
    if _GLUE_CL_EXT is None:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        _GLUE_CL_EXT = load_inline(
            name=f'muse_decode_glue_cl_{arch}', cpp_sources=GLUE_CL_CPP, cuda_sources=GLUE_CL_CUDA,
            functions=['boundary_cl', 'route_cl'],
            extra_cuda_cflags=['-O3', '-std=c++20', '--expt-relaxed-constexpr',
                               f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo'], verbose=False)
    return _GLUE_CL_EXT


def _glue_cl_boundary(branch, stream, alpha, beta, prew, postw, geo):
    p, s = branch
    t = stream.shape[0]
    _, c, nt, rounds = geo
    out = torch.empty((t, 8192), device=stream.device, dtype=torch.bfloat16)
    sout = torch.empty_like(stream)
    _glue_cl_ext().boundary_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1), p.shape[2],
                               stream.data_ptr(), sout.data_ptr(), out.data_ptr(),
                               postw.data_ptr() if postw is not None else 0, alpha.data_ptr(), beta.data_ptr(),
                               prew.data_ptr(), t, c, nt, rounds, torch.cuda.current_stream(stream.device).cuda_stream)
    return out, sout


def _glue_cl_route(pr, state, data, geo):
    p, s = pr
    t = p.shape[1]
    dev = p.device
    _, c, nt, _ = geo
    x = torch.empty((t, 4096), device=dev, dtype=torch.bfloat16)
    ids = torch.empty((t, 8), device=dev, dtype=torch.int32)
    wts = torch.empty((t, 8), device=dev, dtype=torch.float32)
    lids = torch.empty_like(ids)
    _glue_cl_ext().route_cl(p.data_ptr(), int(p.dtype == torch.float32), s, p.stride(0), p.stride(1),
                            state['router_bias'].data_ptr(), state['pre_expert_w'].data_ptr(),
                            data['local_start'].data_ptr(), x.data_ptr(), ids.data_ptr(), wts.data_ptr(),
                            lids.data_ptr(), 0, 0, t, c, nt, torch.cuda.current_stream(dev).cuda_stream)
    return x, ids, wts, lids
'''

OLD_B = "    p,s=branch;t=stream.shape[0];warps,replicas=plan['geometry'][phase]\n"
NEW_B = ("    if plan['geometry'][phase][0]=='cl':return _glue_cl_boundary(branch,stream,alpha,beta,prew,postw,plan['geometry'][phase])\n"
         + OLD_B)
OLD_R = "    p,s=pr;t=p.shape[1];dev=p.device;warps,replicas=plan['geometry']['route']\n"
NEW_R = "    if plan['geometry']['route'][0]=='cl':return _glue_cl_route(pr,state,data,plan['geometry']['route'])\n" + OLD_R
OLD_POOL = "        s5_large_future=native_pool.submit(pm_module)\n"
NEW_POOL = OLD_POOL + "        glue_cl_future=native_pool.submit(_glue_cl_ext)\n"
OLD_JOIN = "        state['s5_large_mod']=s5_large_future.result()\n"
NEW_JOIN = OLD_JOIN + "        glue_cl_future.result()\n"


def _sub(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'expected one {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def patch(src: str, rows=None) -> str:
    rows = ROWS if rows is None else rows
    cuda = (HERE / 'glue_cl.cu').read_text()
    anchor = src.index('\n', src.index('BEST7_DENSE_CHAIN = {'))
    anchor = src.index('\n}\n', anchor) + 3  # end of the BEST7_DENSE_CHAIN literal
    block = BLOCK.replace('__ROWS__', repr(rows)).replace('__CUDA__', repr(cuda)).replace('__CPP__', repr(CPP))
    src = src[:anchor] + block + src[anchor:]
    for old, new in ((OLD_B, NEW_B), (OLD_R, NEW_R), (OLD_POOL, NEW_POOL), (OLD_JOIN, NEW_JOIN)):
        src = _sub(src, old, new)
    return src


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
