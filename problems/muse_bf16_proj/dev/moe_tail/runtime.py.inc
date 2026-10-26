MT_MODULES=[];MT_PMZ_MODULES=[]
def mt_module():
    if not MT_MODULES:
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nvoid mt_route(std::vector<int64_t>,int64_t,int64_t,int64_t);void mt_finish(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> mt_info(int64_t);'
        MT_MODULES.append(load_inline(name='p7_tail_'+hashlib.sha256(MT_CUDA.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=MT_CUDA,
            functions=['mt_route','mt_finish','mt_info'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False))
    return MT_MODULES[0]
def mt_pmz_module():
    if not MT_PMZ_MODULES:
        arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
        cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <cstdint>\n#include <vector>\nint64_t pmz_init();int64_t pmz_smem(int64_t,int64_t);int64_t pmz_resident(int64_t,int64_t,int64_t);void pmz_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
        mod=load_inline(name='p7_pmz_'+hashlib.sha256(MT_PMZ_CUDA.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=MT_PMZ_CUDA,
            functions=['pmz_init','pmz_smem','pmz_resident','pmz_run'],no_implicit_headers=True,with_pytorch_error_handling=False,
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
        mod.pmz_init();MT_PMZ_MODULES.append(mod)
    return MT_PMZ_MODULES[0]
