#include <torch/extension.h>
#include "./lib/sputnik_spops.cpp"
#include "./lib/sputnik_v2_spops.cpp"
#include "./lib/shuffler_spops.cpp"
#include <vector>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("csr_add_fp32", &csr_add_fp32, "CSR Add FP32 (CUDA)");
  m.def("csr_add_fp16", &csr_add_fp16, "CSR Add FP16 (CUDA)");
  m.def("csr_add_bf16", &csr_add_bf16, "CSR Add BF16 (CUDA)");
  m.def("sputnik_spmm_fp32", &sputnik_spmm_fp32, "Sputnik SpMM FP32 (CUDA)");
  m.def("sputnik_spmm_fp16", &sputnik_spmm_fp16, "Sputnik SpMM FP16 (CUDA)");
  m.def("sputnik_v2_sddmm_fp32", &sputnik_v2_sddmm_fp32, "Sputnik SDDMM FP32 (CUDA)");
  m.def("sputnik_v2_sddmm_fp32_benchmark", &sputnik_v2_sddmm_fp32_benchmark, "Sputnik SDDMM FP32 (CUDA)");
  m.def("sputnik_sddmm_fp32", &sputnik_sddmm_fp32, "Sputnik SDDMM FP32 (CUDA)");
  m.def("sputnik_sddmm_fp32", &sputnik_sddmm_fp32_benchmark, "Sputnik SDDMM FP32 (CUDA)");
  m.def("csr_transpose_fp32", &csr_transpose_fp32, "cuSparse CSR Transpose FP32 (CUDA)");
  m.def("csr_transpose_fp16", &csr_transpose_fp16, "cuSparse CSR Transpose FP16 (CUDA)");
}