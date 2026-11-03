#define DC2_DECODE 5

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define IFD __device__ __forceinline__

IFD int4 r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

IFD uint4 r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec_old(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(T0, T1, cm);
  const uint32_t hhi = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(T0, T1, cm >> 16);
  a[0] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mul(uint32_t a, uint32_t scale) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(scale));
  return d;
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mask_bias(uint32_t a) {
  uint32_t d;
  asm("lop3.b32 %0, %1, 0x87ff87ff, 0x40004000, 0xea;" : "=r"(d) : "r"(a));
  return d;
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L0, low_high, 0x5140u);
  a[1] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L0, low_high, 0x7362u);
  a[2] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L1, high_high, 0x5140u);
  a[3] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_prmt(L1, high_high, 0x7362u);
}

__device__ __noinline__ void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_never(p);
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

IFD uint32_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

IFD bool r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

IFD uint64_t r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_wait_bar(uint32_t bar) {
  if (r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_try(bar)) return;
  const uint64_t t0 = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gtime();
  while (!r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_try(bar))
    if (r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gtime() - t0 > 20000000ull) __trap();
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

IFD float4 r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(float* p, float v) { *p = v; }

template <int FMT> struct M32xQFmt;
template <> struct M32xQFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32xQFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32xQFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32xQFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_decode(const int4 (&w)[M32xQFmt<FMT>::NL], uint32_t (&A)[M32xQFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32xQFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32xQFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
IFD void r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
__global__ void __launch_bounds__(NTMAX,(NTMAX==128?6:3)) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32xQFmt<FMT>::NL, NA = M32xQFmt<FMT>::NA, MTT = MT * TG;
  constexpr bool NOW = DIAG & 256;
  constexpr bool C12 = (FMT == 1 || FMT == 3) && !NOW;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED >= 2 && RED <= 6) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  const int wbytes = rawbytes + (XM == 4 ? CWR * CWK * RT * NA * 2048 : 0);
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_rank();
      if constexpr(RED >= 5) {
        if(tid == 0) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_init(bar, 1);
        __syncthreads();
      }
      if ((RED >= 5 || rank == 0) && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        if constexpr(RED < 5) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_init(bar, 1);
        if constexpr(RED >= 5) {
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_expect(bar, (S-1)*(RED==5?NW:WR)*RT*MTT*512u/S);
        } else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_fence_init();
      }
      r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4* rawring = reinterpret_cast<int4*>(smem + xbytes);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0 && !NOW) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0 && !NOW) if (XM != 4 || tg == 0) w[slot][j][l] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp);
          xv[slot][n][1] = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
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
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
          if constexpr (!NOX && (XM == 7 || XM == 8 || XM == 11)) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp), z1=r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_order_point(m, outp);
          }
          if constexpr (CP > 0 && !NOW) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) {
            if constexpr (NOW) {
#pragma unroll
              for (int a=0;a<NA*16;a++) A[j][a]=0x3f803f80u;
            } else if (XM != 4 || tg == 0) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          }
          if constexpr (CP > 0 && !NOW) stage_weights(si%CP,si+CP);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp); z1 = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (CP > 0 && !NOW) asm volatile("cp.async.wait_group 0;" ::: "memory");
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
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(out, 1.f);
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
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED >= 5) {
          const int NI = WR*RT*MTT*32;
          const int OWN = NI/S;
          float4* recv = reinterpret_cast<float4*>(smem+xbytes+wbytes);
          auto emit_owned = [&](int idx, float4 v) {
            emit_full(rank*OWN+idx,v);
          };
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_wait();
          if constexpr(RED==5) {
#pragma unroll
            for(int j=0;j<RT;j++)
#pragma unroll
              for(int n=0;n<MT;n++) {
                const int global_idx=((wr*RT+j)*MTT+tg*MT+n)*32+lane;
                const int dst_rank=global_idx/OWN, idx=global_idx%OWN;
                float4* dst=recv+(rank*WK+wk)*OWN+idx;
                const float4 v=make_float4(r[j][n][0],r[j][n][1],r[j][n][2],r[j][n][3]);
                if(rank==dst_rank) *dst=v;
                else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_st_async4(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(dst),dst_rank),v,r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(bar,dst_rank));
              }
          } else {
            float4* part=recv+S*OWN;
#pragma unroll
            for(int j=0;j<RT;j++)
#pragma unroll
              for(int n=0;n<MT;n++)
                part[wk*NI+((wr*RT+j)*MTT+tg*MT+n)*32+lane]=make_float4(r[j][n][0],r[j][n][1],r[j][n][2],r[j][n][3]);
            __syncthreads();
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(part,NI,WK,NI,tid,nt,[&](int idx,float4 v) {
              const int ln=idx&31,rest=idx>>5,token_tile=rest%MTT,tile=rest/MTT;
              const int dst_rank=token_tile%S;
              const int owned_idx=(tile*(MTT/S)+token_tile/S)*32+ln;
              float4* dst=recv+rank*OWN+owned_idx;
              if(rank==dst_rank) *dst=v;
              else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_st_async4(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(dst),dst_rank),v,r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(bar,dst_rank));
            });
          }
          __syncthreads();
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_wait_bar(bar);
          r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(recv,OWN,S*(RED==5?WK:1),OWN,tid,nt,emit_owned);
        } else if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_st_async4(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_wait_bar(bar);
            if constexpr (RED == 4) r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_cl_wait();
            const uint32_t rbar = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(bar, 0);
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_st_async4(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_mapa(r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_wait_bar(bar);
            r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG, int CP>
static int r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  if (RED>=5 && (CWR*RT*MT*TG*32)%S) return -14;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 5) sm = (size_t)NW*RT*MT*TG*512;
    else if (RED == 6) sm = (size_t)(1+WK)*WR*RT*MT*TG*512;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32xQFmt<FMT>::NA * 2048;
  sm += (size_t)CP * NW * RT * M32xQFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
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
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
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

int r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
case 0:return r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch_t<1,2,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
case 1:return r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch_t<0,2,1,1,5,0,128,0,2304,8192,1,4,8,4,7,1,0>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
    default: return -99;
  }
}
