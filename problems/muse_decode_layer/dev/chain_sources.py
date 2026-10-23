"""Mechanical PDL integration in the vendored MoE and attention sources."""

def chained(base):
    a = base.index('mv4_CUDA_SRC = r"""')
    b = base.index('\n\nCUDA_SRC = r"""',a)
    moe=base[a:b]
    moe=moe.replace('int M, int local_start, int nstage) {', 'int M, int local_start, int nstage, bool use_pdl) {')
    moe=moe.replace('  int bar_gen0 = 0;', '''  int bar_gen0 = 0;
#if __CUDA_ARCH__ >= 900
  if (use_pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif''')
    moe=moe.replace('  for (int i = tid; i < npairs; i += NTHR) {', '''#if __CUDA_ARCH__ >= 900
  if (use_pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  for (int i = tid; i < npairs; i += NTHR) {''',1)
    moe=moe.replace('int64_t grid)', 'int64_t grid, bool use_pdl)')
    marker='#define LAUNCH(NTV)'
    ia=moe.index(marker);ib=moe.index('\n  if (nt == 1)',ia)
    moe=moe[:ia]+'''  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(grid); cfg.blockDim=dim3(NTHR); cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaLaunchAttribute attr{};
  if (use_pdl) {
    attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed=1;
    cfg.attrs=&attr; cfg.numAttrs=1;
  }
#define LAUNCH(NTV) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, moe_fused<NTV>, xp, ids.data_ptr<int>(), tw.data_ptr<float>(), pp, \\
      w13p.data_ptr<uint8_t>(), w2p.data_ptr<uint8_t>(), gs13.data_ptr<float>(), gs2.data_ptr<float>(), ip, \\
      y.data_ptr<float>(), ss.data_ptr<float>(), cnt.data_ptr<int>(), op, M, static_cast<int>(local_start), \\
      static_cast<int>(nstage), use_pdl))'''+moe[ib:]
    moe=moe.replace('nt, nstage, state["grid"])','nt, nstage, state["grid"], PDL)')
    moe=moe.replace('state[\"grid\"] * mod.resident_blocks(nt, nstage, M))','state[\"grid\"] * mod.resident_blocks(nt, nstage, M), PDL)')
    base=base[:a]+moe+base[b:]
    a=base.index('\nCUDA_SRC = r"""');b=base.index('\nCPP_SRC = r"""',a)
    attn=base[a:b]
    attn=attn.replace('#include <stdint.h>', '#include <stdint.h>\n#include <stdexcept>')
    for sig in ['attn_main(const Params p) {','attn_combine(const Params p) {']:
        attn=attn.replace(sig,sig+'''
#if __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif''')
    ia=attn.index('  dim3 grid(p.ns,');ib=attn.index('\n}\n\n}  // namespace',ia)
    attn=attn[:ia]+'''  cudaLaunchConfig_t cfg{};
  cudaLaunchAttribute pdl_attr{};
  pdl_attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl_attr.val.programmaticStreamSerializationAllowed=1;
  cfg.attrs=&pdl_attr; cfg.numAttrs=1;
  cfg.gridDim=dim3(p.ns,p.hkv/2,p.batch); cfg.blockDim=dim3(NTHR);
  cfg.dynamicSmemBytes=smem; cfg.stream=stream;
  cudaError_t err=cudaLaunchKernelEx(&cfg,attn_main<L,STAGES>,p);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  const int warps=p.batch*L*p.hq;
  cfg.gridDim=dim3((warps+3)/4); cfg.blockDim=dim3(128); cfg.dynamicSmemBytes=0;
  err=cudaLaunchKernelEx(&cfg,attn_combine<L>,p);
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));'''+attn[ib:]
    base=base[:a]+attn+base[b:]
    # Avoid colliding with the previously vendored extension's module name in one process.
    base=base.replace('muse_decode_attn_split_{arch}', 'muse_decode_attn_pdl_{arch}')
    return base
