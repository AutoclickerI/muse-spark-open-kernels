
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define SM3D __device__ __forceinline__

SM3D int4 v18s_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint4 v18s_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint32_t v18s_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = v18s_prmt(T0, T1, cm);
  const uint32_t hhi = v18s_prmt(T0, T1, cm >> 16);
  a[0] = v18s_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = v18s_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = v18s_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = v18s_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SM3D void v18s_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);
}

SM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SM3D uint32_t v18s_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SM3D bool v18s_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SM3D uint64_t v18s_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
SM3D void v18s_wait_bar(uint32_t bar) {
  if (v18s_mbar_try(bar)) return;
  const uint64_t t0 = v18s_gtime();
  while (!v18s_mbar_try(bar))
    if (v18s_gtime() - t0 > 20000000ull) __trap();
}

SM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SM3D float4 v18s_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SM3D void v18s_store(float* p, float v) { *p = v; }

template <int FMT> struct Sm3Fmt;
template <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) v18s_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = v18s_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = v18s_ldx(xp);
          xv[slot][n][1] = v18s_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            v18s_cl_wait();
            if (t < TV) {
              const uint32_t rbar = v18s_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            v18s_cl_wait();
            const uint32_t rbar = v18s_mapa(bar, 0);
            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int v18s_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
static int v18s_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = v18s_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, OPT>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = (OPT & 8) ? 4 * MT : std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = v18s_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[1];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
@@CASES@@
    default: return -99;
  }
}

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int M, int NT, int ST>
__global__ void v18s_router_scalar(const float* __restrict__ w, const __nv_bfloat16* __restrict__ x,
                                float* __restrict__ y, int m) {
  constexpr int K=8192;
  int tid=threadIdx.x, row=blockIdx.x, lane=tid&31, warp=tid>>5;
  float acc[M][4]={};
  float4 ww[ST];
  uint2 xx[ST][M];
  auto ld=[&](int slot,int i) {
    ww[slot]=reinterpret_cast<const float4*>(w+row*K)[i];
    #pragma unroll
    for(int t=0;t<M;t++) {
      if(t<m) xx[slot][t]=reinterpret_cast<const uint2*>(x+t*K)[i];
    }
  };
  #pragma unroll
  for(int s=0;s<ST;s++) ld(s,tid+s*NT);
  #pragma unroll 1
  for(int base=tid;base<K/4;base+=ST*NT) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      float4 v=ww[s];
      uint2 a[M];
      #pragma unroll
      for(int t=0;t<M;t++) a[t]=xx[s][t];
      if(base+(s+ST)*NT<K/4) ld(s,base+(s+ST)*NT);
      #pragma unroll
      for(int t=0;t<M;t++) if(t<m) {
        __nv_bfloat162 z0=*reinterpret_cast<__nv_bfloat162*>(&a[t].x);
        __nv_bfloat162 z1=*reinterpret_cast<__nv_bfloat162*>(&a[t].y);
        float2 f0=__bfloat1622float2(z0),f1=__bfloat1622float2(z1);
        acc[t][0]=fmaf(v.x,f0.x,acc[t][0]);
        acc[t][1]=fmaf(v.y,f0.y,acc[t][1]);
        acc[t][2]=fmaf(v.z,f1.x,acc[t][2]);
        acc[t][3]=fmaf(v.w,f1.y,acc[t][3]);
      }
    }
  }
  __shared__ float sums[M][NT/32];
  #pragma unroll
  for(int t=0;t<M;t++) if(t<m) {
    float v=(acc[t][0]+acc[t][1])+(acc[t][2]+acc[t][3]);
    #pragma unroll
    for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
    if(lane==0) sums[t][warp]=v;
  }
  __syncthreads();
  if(warp==0) {
    #pragma unroll
    for(int t=0;t<M;t++) if(t<m) {
      float v=lane<NT/32?sums[t][lane]:0.f;
      #pragma unroll
      for(int d=16;d;d>>=1) v+=__shfl_down_sync(0xffffffff,v,d);
      if(lane==0) y[t*256+row]=v;
    }
  }
}

int v18s_router_launch(int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query) {
  auto kern=v18s_router_scalar<1,512,2>;
  if(query==2) {
    cudaFuncAttributes fa;
    if(cudaFuncGetAttributes(&fa,kern)!=cudaSuccess) return -10;
    return fa.numRegs*1000+(int)std::min<size_t>(fa.localSizeBytes,999);
  }
  if(m!=1) return -5;
  kern<<<256,512,0,(cudaStream_t)stream>>>((const float*)w,(const __nv_bfloat16*)x,(float*)out,m);
  return cudaGetLastError()==cudaSuccess?0:-8;
}
