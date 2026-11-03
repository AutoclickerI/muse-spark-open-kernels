
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <vector>

#define LT_OK(call) TORCH_CHECK((call) == CUBLAS_STATUS_SUCCESS, "cuBLASLt failure: ", #call)

using Config = std::array<int64_t, 9>;

Config config_of(const cublasLtMatmulAlgo_t& a) {
  Config c{};
  for (int i = 0; i < 9; ++i) {
    size_t written = 0;
    if (i >= 7) {
      uint16_t value = 0;
      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,
                                          &value, sizeof(value), &written);
      c[i] = value;
    } else {
      uint32_t value = 0;
      cublasLtMatmulAlgoConfigGetAttribute(&a, (cublasLtMatmulAlgoConfigAttributes_t)i,
                                          &value, sizeof(value), &written);
      c[i] = value;
    }
  }
  return c;
}

bool set_config(cublasLtMatmulAlgo_t& a, int attr, int value) {
  if (attr >= 7) {
    uint16_t v = value;
    return cublasLtMatmulAlgoConfigSetAttribute(&a,
      (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
  }
  uint32_t v = value;
  return cublasLtMatmulAlgoConfigSetAttribute(&a,
    (cublasLtMatmulAlgoConfigAttributes_t)attr, &v, sizeof(v)) == CUBLAS_STATUS_SUCCESS;
}

std::vector<uint32_t> cap_list(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  size_t bytes = 0;
  cublasLtMatmulAlgoCapGetAttribute(&a, attr, nullptr, 0, &bytes);
  if (!bytes) return {0};
  std::vector<uint32_t> values(bytes / sizeof(uint32_t));
  if (cublasLtMatmulAlgoCapGetAttribute(&a, attr, values.data(), bytes, &bytes) != CUBLAS_STATUS_SUCCESS)
    return {0};
  return values;
}

uint32_t cap_int(const cublasLtMatmulAlgo_t& a, cublasLtMatmulAlgoCapAttributes_t attr) {
  uint32_t v = 0;
  size_t written = 0;
  cublasLtMatmulAlgoCapGetAttribute(&a, attr, &v, sizeof(v), &written);
  return v;
}

struct LtContext {
  cublasLtHandle_t handle = nullptr;
  torch::Tensor workspace;
  explicit LtContext(torch::Tensor ws) : workspace(ws) {
    c10::cuda::CUDAGuard guard(ws.device());
    LT_OK(cublasLtCreate(&handle));
  }
  ~LtContext() { if (handle) cublasLtDestroy(handle); }
};

struct LtPlan {
  std::shared_ptr<LtContext> context;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr, b_layout = nullptr, d_layout = nullptr;
  torch::Tensor weight, bias;
  int m, n, k, layout, epilogue;
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  std::vector<size_t> workspaces;
  std::set<Config> seen;
  std::vector<std::vector<int64_t>> diagnostics;
  int checked = 0, heuristic_count = 0, id_count = 0;

  LtPlan(std::shared_ptr<LtContext> ctx, torch::Tensor w, int rows, int cols,
         int inner, int variant, int epi)
      : context(ctx), weight(w), m(rows), n(cols), k(inner), layout(variant), epilogue(epi) {
    c10::cuda::CUDAGuard guard(w.device());
    LT_OK(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    int8_t fast = 0;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast, sizeof(fast)));
    bool is_row = layout % 2;
    bool is_packed = layout >= 2;
    cublasOperation_t trans_a = !is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t trans_b = is_row && !is_packed ? CUBLAS_OP_T : CUBLAS_OP_N;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
    if (is_row) {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, m, k, k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, is_packed ? k : n,
                                      is_packed ? n : k, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, m, n, n));
      cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
      for (auto d : {a_layout, b_layout, d_layout})
        LT_OK(cublasLtMatrixLayoutSetAttribute(d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    } else {
      LT_OK(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16BF, is_packed ? n : k,
                                      is_packed ? k : n, is_packed ? n : k));
      LT_OK(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16BF, k, m, k));
      LT_OK(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_16BF, n, m, n));
    }
    cublasLtEpilogue_t e = epi ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &e, sizeof(e)));
    if (epi) {
      bias = torch::zeros({std::max(m, n)}, w.options());
      void* ptr = bias.data_ptr();
      LT_OK(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &ptr, sizeof(ptr)));
    }
  }

  ~LtPlan() {
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
    if (d_layout) cublasLtMatrixLayoutDestroy(d_layout);
  }

  bool add(const cublasLtMatmulAlgo_t& algo) {
    Config c = config_of(algo);
    if (seen.count(c)) return false;
    seen.insert(c);
    ++checked;
    if (diagnostics.size() < 32) {
      uint64_t flags = 0;
      size_t written = 0;
      int status = cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
        &flags, sizeof(flags), &written);
      std::vector<int64_t> row(c.begin(), c.end());
      row.push_back(flags);
      row.push_back(status);
      diagnostics.push_back(row);
    }
    if (c[3] != CUBLASLT_REDUCTION_SCHEME_NONE && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)
      return false;
    if (c[2] > 1 && c[3] != CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE) return false;
    uint64_t flags = 0;
    size_t written = 0;
    if (cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
                                         &flags, sizeof(flags), &written) != CUBLAS_STATUS_SUCCESS) return false;
    if ((flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) != CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
      return false;
    // Algo 66 reports INPUT_16F even for CUDA_R_16BF descriptors in cuBLASLt 13.
    // The descriptors fix BF16 storage; reject TF32 and require FP32 accumulators.
    if (flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_INPUT_TF32)
      return false;
    cublasLtMatmulHeuristicResult_t result{};
    if (cublasLtMatmulAlgoCheck(context->handle, operation, a_layout, b_layout,
                               d_layout, d_layout, &algo, &result) != CUBLAS_STATUS_SUCCESS) return false;
    if (result.state != CUBLAS_STATUS_SUCCESS || result.workspaceSize > size_t(context->workspace.numel())) return false;
    algorithms.push_back(algo);
    workspaces.push_back(result.workspaceSize);
    return true;
  }

  int restore(std::vector<int64_t> values) {
    TORCH_CHECK(values.size() == 9, "expected nine cuBLASLt configuration fields");
    cublasLtMatmulAlgo_t algo{};
    if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, values[0], &algo) != CUBLAS_STATUS_SUCCESS) return -1;
    for (int i = 1; i < 9; ++i) {
      if (!set_config(algo, i, values[i]) && values[i] != 0) return -1;
    }
    if (!add(algo)) return -1;
    return algorithms.size() - 1;
  }

  void enumerate(bool exhaustive) {
    c10::cuda::CUDAGuard guard(weight.device());
    cublasLtMatmulPreference_t pref;
    LT_OK(cublasLtMatmulPreferenceCreate(&pref));
    uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
    LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                             &reduction, sizeof(reduction)));
    for (size_t ws : {size_t(0), size_t(4 << 20), size_t(32 << 20), size_t(128 << 20)}) {
      ws = std::min(ws, size_t(context->workspace.numel()));
      LT_OK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
      std::vector<cublasLtMatmulHeuristicResult_t> result(256);
      int count = 0;
      auto rc = cublasLtMatmulAlgoGetHeuristic(context->handle, operation, a_layout, b_layout,
                                               d_layout, d_layout, pref, 256, result.data(), &count);
      if (rc != CUBLAS_STATUS_SUCCESS) continue;
      heuristic_count += count;
      for (int i = 0; i < count; ++i) if (result[i].state == CUBLAS_STATUS_SUCCESS) add(result[i].algo);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    if (!exhaustive) return;
    std::array<int, 256> ids{};
    LT_OK(cublasLtMatmulAlgoGetIds(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
      CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids.size(), ids.data(), &id_count));
    for (int i = 0; i < id_count; ++i) {
      cublasLtMatmulAlgo_t a{};
      if (cublasLtMatmulAlgoInit(context->handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
          CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[i], &a) != CUBLAS_STATUS_SUCCESS) continue;
      auto tiles = cap_list(a, CUBLASLT_ALGO_CAP_TILE_IDS);
      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);
      for (auto tile : tiles) for (auto stage : stages) {
        auto b = a;
        if (!set_config(b, 1, tile) || !set_config(b, 6, stage)) continue;
        add(b);
        if (tiles.size() > 100) {
          for (int cluster : {2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {
            auto s = b;
            if (set_config(s, 8, cluster)) add(s);
          }
        }
      }
    }
  }

  void expand(std::vector<int> indices) {
    for (int index : indices) {
      auto a = algorithms.at(index);
      if (cap_int(a, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT) &&
          (cap_int(a, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK) & CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) {
        for (int split : {2, 3, 4, 6, 8, 12, 16}) {
          auto b = a;
          if (set_config(b, 2, split) && set_config(b, 3, CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE)) add(b);
        }
      }
      auto stages = cap_list(a, CUBLASLT_ALGO_CAP_STAGES_IDS);
      for (auto stage : stages) {
        auto b = a;
        if (set_config(b, 6, stage)) add(b);
      }
      for (int swizzle = 0; swizzle <= int(cap_int(a, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT)); ++swizzle) {
        for (int custom = 0; custom <= int(cap_int(a, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)); ++custom) {
          for (int cluster : {0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12}) {
            auto b = a;
            if (set_config(b, 4, swizzle) && set_config(b, 5, custom) && set_config(b, 8, cluster)) add(b);
          }
        }
      }
    }
  }

  std::vector<std::vector<int64_t>> metadata() const {
    std::vector<std::vector<int64_t>> result;
    for (size_t i = 0; i < algorithms.size(); ++i) {
      Config c = config_of(algorithms[i]);
      std::vector<int64_t> row(c.begin(), c.end());
      row.push_back(workspaces[i]);
      result.push_back(row);
    }
    return result;
  }

  int run(torch::Tensor x, torch::Tensor out, int index) {
    const float alpha = 1.0f, beta = 0.0f;
    const void* a = layout % 2 ? x.data_ptr() : weight.data_ptr();
    const void* b = layout % 2 ? weight.data_ptr() : x.data_ptr();
    return int(cublasLtMatmul(context->handle, operation, &alpha, a, a_layout, b, b_layout,
      &beta, out.data_ptr(), d_layout, out.data_ptr(), d_layout, &algorithms.at(index),
      context->workspace.data_ptr(), workspaces.at(index), at::cuda::getCurrentCUDAStream(x.get_device())));
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("version", []() { return cublasLtGetVersion(); });
  pybind11::class_<LtContext, std::shared_ptr<LtContext>>(m, "Context")
    .def(pybind11::init<torch::Tensor>());
  pybind11::class_<LtPlan, std::shared_ptr<LtPlan>>(m, "Plan")
    .def(pybind11::init<std::shared_ptr<LtContext>, torch::Tensor, int, int, int, int, int>())
    .def("enumerate", &LtPlan::enumerate).def("expand", &LtPlan::expand)
    .def("metadata", &LtPlan::metadata).def("restore", &LtPlan::restore).def("run", &LtPlan::run)
    .def_readonly("checked", &LtPlan::checked).def_readonly("heuristic_count", &LtPlan::heuristic_count)
    .def_readonly("diagnostics", &LtPlan::diagnostics)
    .def_readonly("id_count", &LtPlan::id_count);
}
