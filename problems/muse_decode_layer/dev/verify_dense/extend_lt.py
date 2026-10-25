from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'incumbent_lt.cu').read_text().replace('#include <cstdlib>', '#include <cstdlib>\n#include <vector>\n#include <unordered_set>')
needle='  if(overflow)return lean_fail("all Lt option capacity",-1);'
extra=r'''
  // Enumerate every kernel ID, every supported tile ID and stage ID. This
  // reaches kernels that AlgoGetHeuristic never returned for this shape.
  auto cap_list=[&](const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoCapAttributes_t attr){
    size_t bytes=0;std::vector<uint32_t> values;
    if(cublasLtMatmulAlgoCapGetAttribute(&algo,attr,nullptr,0,&bytes)!=CUBLAS_STATUS_SUCCESS)return values;
    values.resize(bytes/sizeof(uint32_t));
    if(bytes && cublasLtMatmulAlgoCapGetAttribute(&algo,attr,values.data(),bytes,&bytes)!=CUBLAS_STATUS_SUCCESS)values.clear();
    return values;
  };
  int ids[1024],num_ids=0;cudaDataType_t output=kind?CUDA_R_32F:CUDA_R_16BF;
  LEAN_LT(cublasLtMatmulAlgoGetIds(lean_lt_handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,
    CUDA_R_16BF,CUDA_R_16BF,output,output,1024,ids,&num_ids));
  for(int ii=0;ii<num_ids;++ii){
    cublasLtMatmulAlgo_t base={};
    if(cublasLtMatmulAlgoInit(lean_lt_handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,
       CUDA_R_16BF,CUDA_R_16BF,output,output,ids[ii],&base)!=CUBLAS_STATUS_SUCCESS)continue;
    add(base);
    auto tiles=cap_list(base,CUBLASLT_ALGO_CAP_TILE_IDS);
    auto stages=cap_list(base,CUBLASLT_ALGO_CAP_STAGES_IDS);
    if(stages.empty())stages.push_back(0);
    for(auto tile:tiles)for(auto stage:stages){
      auto config=base;
      if(cublasLtMatmulAlgoConfigSetAttribute(&config,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS)continue;
      if(cublasLtMatmulAlgoConfigSetAttribute(&config,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stage,sizeof(stage))!=CUBLAS_STATUS_SUCCESS)continue;
      add(config);
    }
  }
'''
assert needle in s;s=s.replace(needle,extra+needle)
s=s.replace('  auto add=[&](const cublasLtMatmulAlgo_t& algo) {', '  std::unordered_set<std::string> seen;\n  auto add=[&](const cublasLtMatmulAlgo_t& algo) {')
s=s.replace('    for(int i=0;i<valid_count;++i)if(std::memcmp(valid+i,&algo,sizeof(algo))==0)return;', '    if(!seen.emplace((const char*)&algo,sizeof(algo)).second)return;')
s+=r'''
extern "C" int vd_lt_info(const void* codes,int index,int* out){
  auto* algo=(const cublasLtMatmulAlgo_t*)codes+index;size_t wrote=0;
  cublasLtMatmulAlgoConfigAttributes_t fields[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,CUBLASLT_ALGO_CONFIG_STAGES_ID,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
  for(int i=0;i<7;++i){out[i]=-1;cublasLtMatmulAlgoConfigGetAttribute(algo,fields[i],out+i,sizeof(int),&wrote);}
  return 0;
}
'''
(HERE/'lt.cu').write_text(s)
