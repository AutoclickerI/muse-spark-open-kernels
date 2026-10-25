"""CUDA sources of the branch/remainder labs, derived from the frozen S6 candidate text."""
import ast
import hashlib
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[4]
S6 = R / 'problems/muse_decode_layer/cand_r3_best7_s6.py'
S6_SHA = '4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f'


def s6_text():
    text = S6.read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == S6_SHA
    return text


def s6_value(text, key):
    return next(ast.literal_eval(n.value) for n in ast.parse(text).body
                if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == key for t in n.targets))


KINDS256 = (  # id, NT, FIRST, VEC, FM, LO, HI
    (0, 16, 1, 1, 1, 0, 16), (1, 32, 1, 1, 1, 16, 'mwbr::BIG'),
    (2, 16, 0, 0, 1, 0, 16), (3, 32, 0, 0, 1, 16, 'mwbr::BIG'),
    (4, 32, 1, 1, 2, 16, 32), (5, 16, 1, 1, 2, 0, 16), (6, 16, 1, 1, 2, 8, 16), (7, 8, 1, 1, 2, 0, 8),
    (8, 32, 0, 0, 2, 16, 32), (9, 16, 0, 0, 2, 0, 16), (10, 16, 0, 0, 2, 8, 16), (11, 8, 0, 0, 2, 0, 8),
    (12, 32, 0, 0, 0, 0, 0))
KINDS_HH = (  # on H_H's buffers: 0/1 and 4/5/8/9 run the heavy-only T32 list, 12 the 8<count<=16 T8 list;
    # W2 keeps H_H's wide copies (VEC)
    (0, 32, 1, 1, 0, 0, 0), (1, 32, 0, 1, 0, 0, 0),
    (4, 32, 1, 1, 2, 16, 32), (5, 16, 1, 1, 2, 0, 16), (8, 32, 0, 1, 2, 16, 32), (9, 16, 0, 1, 2, 0, 16),
    (12, 16, 0, 1, 1, 8, 16),
    (20, 32, 1, 2, 0, 0, 0), (21, 32, 0, 2, 0, 0, 0),  # 20/21 = 0/1 with the lean K loop (VEC==2)
    (22, 32, 1, 2, 1, 16, 32), (23, 32, 0, 2, 1, 16, 32),  # lean N32 for 16<count<=32 (T32 list)
    (24, 64, 1, 2, 0, 0, 0), (25, 64, 0, 2, 0, 0, 0),  # lean N64 on the count>32 T64 list: one decode per 64 rows
    (26, 32, 1, 3, 0, 0, 0))  # 20 with gate/up rows interleaved by 8 in prepare: SwiGLU in registers
KINDS_S6 = ((4, 32, 1, 1, 2, 16, 32), (5, 16, 1, 1, 2, 0, 16), (8, 32, 0, 0, 2, 16, 32), (9, 16, 0, 0, 2, 0, 16))  # R_full of 110599ba


def kinds_macro(kinds):
    b = lambda v: 'true' if v else 'false'
    rows = ' '.join(f'F({i},{nt},{b(f)},{int(v)},{fm},{lo},{hi})' for i, nt, f, v, fm, lo, hi in kinds)
    return '#define BR_KINDS(F) ' + rows + '\n'


def pmz_source(text):
    """H_H's NT1 phase kernel, renamed pmz, plus ZSS: the small W13 zeroes its experts' SS rows
    (m==0 unit of each token block) so the route kernel no longer clears cap*64 floats; THR is the
    small-expert count limit (16, or 8 for a W2 that sends 8<count<=16 to N16)."""
    cuda = s6_value(text, 'S5_HH_CUDA')
    a = cuda.index('namespace pmnt {')
    b = cuda.index('}  // namespace', a) + len('}  // namespace')
    body = cuda[a:b].replace('namespace pmnt {', 'namespace pmz {')
    old = 'template<int NT,bool FIRST,bool SMALL>\n__global__ void __launch_bounds__(NTHR,2) phase('
    assert body.count(old) == 1
    body = body.replace(old, 'template<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\n__global__ void __launch_bounds__(NTHR,2) phase(')
    old = 'if constexpr(SMALL)if(n>16)nb=0;'
    assert body.count(old) == 1
    body = body.replace(old, 'if constexpr(SMALL)if(n>THR)nb=0;')
    anchor = '      } else {\n        const float sc = gs2[e] * 256.f;'
    assert body.count(anchor) == 1
    zss = ('        if constexpr (ZSS) {\n'
           '          if (m == 0) {\n'
           '            for (int z = tid; z < NB * 64; z += NTHR) {\n'
           '              const int zi = nb * NB + (z >> 6);\n'
           '              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\n'
           '            }\n'
           '          }\n'
           '        }\n')
    body = body.replace(anchor, zss + anchor)
    host = r'''
int64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};
 set.template operator()<true,true,16>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}
int64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}
int64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;
 if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));
 else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}
// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.
void pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){
 BR_REQUIRE(v.size()==12);BR_REQUIRE(first?thr==16:(thr==16||thr==8));
 auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(
 (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};
 if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();
 BR_CUDA_CHECK(cudaGetLastError());}
'''
    return body + '\n' + host


def sources():
    text = s6_text()
    gemm = (H / 'gemm.cu.inc').read_text()
    cu_s6 = kinds_macro(KINDS_S6) + gemm
    cu_hh = kinds_macro(KINDS_HH) + gemm + '\n' + pmz_source(text)
    decl = ('#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t br_init();std::vector<int64_t> br_info(int64_t);'
            'void br_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);'
            'void br_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);'
            'void br_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);')
    decl_hh = decl + ('int64_t pmz_init();int64_t pmz_smem(int64_t,int64_t);int64_t pmz_resident(int64_t,int64_t,int64_t);'
                      'void pmz_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);')
    return dict(cu_s6=cu_s6, cpp_s6=decl, cu_hh=cu_hh, cpp_hh=decl_hh)


if __name__ == '__main__':
    s = sources()
    for k in ('cu_s6', 'cu_hh'):
        (H / f'{k}.cu').write_text('#include <cstdint>\n' + s[k])
        print(k, len(s[k]), hashlib.sha256(s[k].encode()).hexdigest()[:16])
