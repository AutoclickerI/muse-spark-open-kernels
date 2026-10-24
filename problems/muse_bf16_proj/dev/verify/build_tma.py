"""Generate a persistent/cluster SS-WGMMA projection block from the validated TMA reader."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
INSTANCES=[(64,1,1,4,1),(64,1,1,4,2),
    (64,1,1,5,1),(64,1,1,5,2),(64,1,1,6,1),(64,1,1,6,2),
    (128,1,1,4,1),(128,1,1,4,2),(128,1,1,6,1),(128,1,1,6,2),
    (256,1,1,3,1),(256,1,1,3,2),(256,1,1,4,1),(256,1,1,4,2),(256,1,1,5,1),(256,1,1,5,2),
    (64,1,2,4,1),(64,1,2,4,2),(128,1,2,3,1),(128,1,2,3,2),
    (128,1,2,4,1),(128,1,2,4,2),(256,1,2,3,1)]


def source():
    spec=importlib.util.spec_from_file_location('wg_source_base',ROOT/'problems/muse_bf16_proj/dev/wg_src.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    s=old.wg_source(INSTANCES)
    def replace(a,b):
        nonlocal s
        assert a in s,a[:100]
        s=s.replace(a,b)
    replace('if (++tries > (uint32_t)MP_SPIN_LIMIT) {','if (false) {')
    replace('mbarrier.try_wait.parity.shared::cta.b64',
            'mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64')
    replace('namespace muse_wg {','''namespace muse_wg {
__device__ __forceinline__ void expect_only(uint64_t* bar, uint32_t bytes) {
  const uint32_t address = (uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
      :: "r"(address), "r"(bytes) : "memory");
}
''')
    # Every multicast receiver publishes its transaction expectation before ANY
    # producer can reuse the slot. The initial expectations precede cluster sync;
    # later expectations precede release of the MC-wide empty barrier.
    replace('mbar_init(&full[s], 1);','mbar_init(&full[s], 1); expect_only(&full[s], STAGE_BYTES);')
    replace('mbar_expect_tx(&full[stage], STAGE_BYTES);','mbar_arrive(&full[stage]);')
    replace('128 * (NCW + 1)','128 * NCW + 32')
    replace('''          const uint16_t mask = (uint16_t)(((1u << MC) - 1) << (MC * blockIdx.y));
          tma_load_2d_mc(sa + A_BYTES + cx * (BT / MC) * 128, &tmX, &full[stage], kc % kx, tok0 + cx * (BT / MC),
                         mask);''','''#ifdef MP_SM90A
          const uint16_t mask = (uint16_t)(((1u << MC) - 1) << (MC * blockIdx.y));
          tma_load_2d_mc(sa + A_BYTES + cx * (BT / MC) * 128, &tmX, &full[stage], kc % kx, tok0 + cx * (BT / MC),
                         mask);
#else
          // Portable transport stand-in: native multicast is optimized for sm90a.
          // Keep identical barriers, cluster scheduling and arithmetic locally.
          for (int part = 0; part < MC; ++part)
            tma_load_2d(sa + A_BYTES + part * (BT / MC) * 128, &tmX, &full[stage], kc % kx,
                        tok0 + part * (BT / MC));
#endif''')
    replace('int nkb, int kx) {','int nkb, int kx, int pipe, int pdl) {')
    replace('  const int row_cta = blockIdx.x * BR;\n  const int tok0 = blockIdx.z * BT;', '''
  if (pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int groups_n = (N / BR) / MC;
  const int jobs = groups_n * ((M + BT - 1) / BT);
  for (int job = blockIdx.x / MC; job < jobs; job += gridDim.x / MC) {
  const int row_cta = (job % groups_n) * MC * BR + (blockIdx.x % MC) * BR;
  const int tok0 = (job / groups_n) * BT;''')
    replace('''        tma_load_2d(sa, &tmW, &full[stage], kc, row_cta);
        if (MC == 1) {''','''        tma_load_2d(sa, &tmW, &full[stage], kc, row_cta);
        if (s == 0 && pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
        if (MC == 1) {''')
    replace('''      wgmma_wait<0>();
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);''','''      if (pipe) wgmma_wait<1>(); else wgmma_wait<0>();
#pragma unroll
      for (int sub = 0; sub < NSUB; ++sub) acc_fence(acc[sub]);''')
    replace('''      if ((threadIdx.x & 127) == 0) {
        mbar_arrive(&empty[stage]);
        if (MC > 1) mbar_arrive_remote(&empty[stage], (uint32_t)(my_rank ^ 1));
      }
    }
  }
  __syncthreads();''','''      if ((threadIdx.x & 127) == 0 && (!pipe || s > 0)) {
        const int release_stage = pipe ? (s + ST - 1) % ST : stage;
        expect_only(&full[release_stage], STAGE_BYTES / NCW);
        mbar_arrive(&empty[release_stage]);
        if (MC > 1) mbar_arrive_remote(&empty[release_stage], (uint32_t)(my_rank ^ 1));
      }
    }
#ifdef MP_SM90A
    wgmma_wait<0>();
#endif
    if (pipe && nsteps > 0 && (threadIdx.x & 127) == 0) {
      const int last = (nsteps - 1) % ST;
      expect_only(&full[last], STAGE_BYTES / NCW);
      mbar_arrive(&empty[last]);
      if (MC > 1) mbar_arrive_remote(&empty[last], (uint32_t)(my_rank ^ 1));
    }
  }
  __syncthreads();''')
    replace('''  if (MC > 1 && ks == 1) cg::this_cluster().sync();  // partner may still arrive on our barriers
#endif
}''','''  if (MC > 1 && ks == 1) cg::this_cluster().sync();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < ST; ++s) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&full[s])) : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(su32(&empty[s])) : "memory");
    }
  }
  }  // next persistent logical tile (all cluster partners use the same schedule)
#endif
}''')
    replace('torch::Tensor& Out, int N, int ks) {','torch::Tensor& Out, int N, int ks, int waves, int pipe, int pdl) {')
    replace('''  cfg.gridDim = dim3(N / BR, ks, (M + BT - 1) / BT);''','''  int dev = 0, sms = 0;
  cudaGetDevice(&dev); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  const int logical = ((N / BR) / MC) * ((M + BT - 1) / BT);
  const int groups = waves ? min(logical, max(1, sms * waves / (MC * ks))) : logical;
  cfg.gridDim = dim3(groups * MC, ks, 1);''')
    replace('cudaLaunchAttribute attr[1];','cudaLaunchAttribute attr[2];')
    replace('''    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  auto err''','''    cfg.numAttrs = 1;
  }
  if (pdl) {
    attr[cfg.numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[cfg.numAttrs].val.programmaticStreamSerializationAllowed = 1;
    ++cfg.numAttrs;
  }
  cfg.attrs = attr;
  auto err''')
    replace('Out.data_ptr(), M, N, KW / 64, KX);','Out.data_ptr(), M, N, KW / 64, KX, pipe, pdl);')
    replace('int64_t ncw, int64_t st, int64_t ks, int64_t mc) {','int64_t ncw, int64_t st, int64_t ks, int64_t mc, int64_t waves, int64_t pipe, bool pdl) {')
    replace('Out, (int)N, (int)ks);','Out, (int)N, (int)ks, (int)waves, (int)pipe, (int)pdl);')
    prefix='''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg = cooperative_groups;
'''
    return prefix+s


def build():
    src=source();(HERE/'tma.cu').write_text(src)
    module=(HERE/'library.py').read_text()+'\n\nVP_TMA_SOURCE='+repr(src)+'\nVP_TMA_INSTANCES='+repr(INSTANCES)+'\n\n'+(HERE/'tma_tail.py').read_text()
    path=HERE/'verify_projection.py';path.write_text(module);return path


if __name__=='__main__':print(build())
