

# ---------------------------------------------------------------------------------------------
# H200 probes (check runs only): achievable HBM stream rate and L2->SM rate, DeepGEMM BF16 GEMM.
PROBE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
template <int U, int L2HIT>
__global__ void probe_kernel(const uint4* __restrict__ p, long long nvec, long long wrap, int iters, uint4* out) {
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)gridDim.x * blockDim.x;
  uint32_t acc = 0;
  for (int it = 0; it < iters; ++it) {
    long long base = tid + (long long)it * stride * U;
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      long long i = base + (long long)u * stride;
      if (L2HIT) i = i & (wrap - 1);
      const uint4* q = p + i;
      if (L2HIT)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(q));
      else
        asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(q));
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678u) out[0] = make_uint4(acc, 0, 0, 0);
}
void probe(torch::Tensor buf, torch::Tensor out, int64_t ctas, int64_t threads, int64_t u, int64_t l2, int64_t total_vec, int64_t wrap) {
  const long long per_iter = ctas * threads * u;
  const int iters = (int)(total_vec / per_iter);
  auto s = at::cuda::getCurrentCUDAStream();
  const uint4* p = (const uint4*)buf.data_ptr();
  uint4* o = (uint4*)out.data_ptr();
#define PC(U_, L_) if (u == U_ && l2 == L_) { probe_kernel<U_, L_><<<ctas, threads, 0, s>>>(p, total_vec, wrap, iters, o); return; }
  PC(4, 0) PC(8, 0) PC(16, 0) PC(4, 1) PC(8, 1)
}
"""


def _probe_mod():
    from torch.utils.cpp_extension import load_inline
    return load_inline(name="muse_proj_probe_v1", cpp_sources="void probe(torch::Tensor buf, torch::Tensor out, int64_t ctas, int64_t threads, int64_t u, int64_t l2, int64_t total_vec, int64_t wrap);",
                       cuda_sources=PROBE_SRC, functions=["probe"], extra_cuda_cflags=["-O3", "-std=c++20"], verbose=False)


def _probes(timer):
    pm = _probe_mod()
    buf = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")  # 64 MB stream
    out = torch.empty(16, dtype=torch.uint8, device="cuda")
    nvec = buf.numel() // 16
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    for ctas_per_sm, threads, u in ((1, 512, 8), (2, 256, 8), (2, 512, 4), (4, 256, 4), (2, 256, 16), (1, 1024, 4)):
        ctas = sms * ctas_per_sm
        total = nvec // (ctas * threads * u) * (ctas * threads * u)
        g = _graph(lambda: pm.probe(buf, out, ctas, threads, u, 0, total, total))
        t = timer.time(g)
        print(f"HBM stream {total * 16 / 1e6:.1f} MB ctas {ctas} thr {threads} u {u}: {t:.2f} us = {total * 16 / t / 1e6:.2f} TB/s")
    for wrap_mb in (4, 16, 32):
        wrap = wrap_mb * 1024 * 1024 // 16
        for ctas_per_sm, threads, u in ((2, 256, 8), (4, 256, 4)):
            ctas = sms * ctas_per_sm
            total = 4 * nvec // (ctas * threads * u) * (ctas * threads * u)  # 256 MB of L2 reads
            g = _graph(lambda: pm.probe(buf, out, ctas, threads, u, 1, total, wrap))
            g.replay()  # warm: the wrapped window now sits in L2
            ts = []
            for _ in range(10):
                timer.e0.record(); g.replay(); timer.e1.record(); timer.e1.synchronize()
                ts.append(timer.e0.elapsed_time(timer.e1) * 1e3)
            t = _st.median(ts) - timer.base
            print(f"L2 read window {wrap_mb} MB, {total * 16 / 1e6:.0f} MB read, ctas {ctas} u {u}: {t:.2f} us = {total * 16 / t / 1e6:.2f} TB/s")
    del buf


def _deepgemm(timer):
    try:
        import deep_gemm
    except Exception as e:  # noqa: BLE001
        print(f"deep_gemm import failed: {e}")
        return
    t0 = _time.time()
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        n, k = _DIMS[layer]
        w = torch.randn(n, k, device="cuda").to(torch.bfloat16)
        for m in (64, 128, 256):
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            try:
                tc = _time.time()
                deep_gemm.bf16_gemm_nt(x, w, d)
                torch.cuda.synchronize()
                tc = _time.time() - tc
                ref = F.linear(x, w)
                err = float((d.float() - ref.float()).norm() / ref.float().norm())
                tdg = timer.time(_graph(lambda: deep_gemm.bf16_gemm_nt(x, w, d)))
                tcb = timer.time(_graph(lambda: F.linear(x, w)))
                print(f"DG {layer[:4]} m{m}: deepgemm {tdg:.2f} cublas {tcb:.2f} us (compile {tc:.1f}s err {err:.1e})", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"DG {layer} m{m} failed: {type(e).__name__}: {e}")
        if _time.time() - t0 > 150:
            print("DG time budget hit")
            break
