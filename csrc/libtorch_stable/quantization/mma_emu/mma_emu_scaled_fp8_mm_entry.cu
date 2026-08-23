#include <cuda_runtime.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>

#include "libtorch_stable/torch_utils.h"

#include "mma_emu_config_check.h"
#include "mma_emu_scaled_fp8_mm_kernels.cuh"

namespace ds = vllm::mma_emu::design_space;

// Emulated FP8 E4M3 scaled mm. Mirrors the cutlass_scaled_mm interface so the
// two are interchangeable behind the linear-kernel registry, with the
// accumulation configuration appended.
void mma_emu_scaled_fp8_mm(torch::stable::Tensor& out,
                           const torch::stable::Tensor& a,
                           const torch::stable::Tensor& b,
                           const torch::stable::Tensor& a_scales,
                           const torch::stable::Tensor& b_scales,
                           const std::optional<torch::stable::Tensor>& bias,
                           int64_t algorithm, int64_t f_bits, int64_t g_bits,
                           int64_t group_size, int64_t chunk_size) {
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2);
  STD_TORCH_CHECK(out.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == out.size(1));

  STD_TORCH_CHECK(a.stride(1) == 1 && out.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                        // Column-major

  // The kernel indexes with a packed stride, so a sliced or padded view would
  // be read as though it were dense, silently returning the wrong product
  // rather than failing. Require the packed layout instead.
  STD_TORCH_CHECK(a.stride(0) == a.size(1),
                  "MMA-Emu scaled_fp8_mm: a must be contiguous");
  STD_TORCH_CHECK(out.stride(0) == out.size(1),
                  "MMA-Emu scaled_fp8_mm: out must be contiguous");
  STD_TORCH_CHECK(b.stride(1) == b.size(0),
                  "MMA-Emu scaled_fp8_mm: b must be contiguous");

  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  // Per-tensor scales only: the emulation applies a single scale in the
  // epilogue rather than a per-row/column vector.
  STD_TORCH_CHECK(
      a_scales.numel() == 1 && b_scales.numel() == 1,
      "MMA-Emu scaled_fp8_mm: only per-tensor scales are supported");
  STD_TORCH_CHECK(a_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(b_scales.scalar_type() ==
                  torch::headeronly::ScalarType::Float);

  const auto out_dtype = out.scalar_type();
  STD_TORCH_CHECK(out_dtype == torch::headeronly::ScalarType::BFloat16 ||
                      out_dtype == torch::headeronly::ScalarType::Half,
                  "MMA-Emu scaled_fp8_mm: out must be BFloat16 or Half");

  if (bias) {
    STD_TORCH_CHECK(
        bias->numel() == b.size(1) && bias->is_contiguous() && bias->dim() == 1,
        "MMA-Emu scaled_fp8_mm: bias must be 1D with N elements");
    STD_TORCH_CHECK(bias->scalar_type() == out_dtype,
                    "MMA-Emu scaled_fp8_mm: bias dtype must match out");
  }

  // Backstop: the Python kernel selector rejects a bad configuration earlier
  // and with the same wording, but this operator is reachable directly.
  const std::string err = vllm::mma_emu::mma_emu_config_error(
      algorithm, f_bits, g_bits, group_size, chunk_size, /*is_fp4=*/false);
  STD_TORCH_CHECK(err.empty(), "MMA-Emu scaled_fp8_mm: ", err);

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(a.get_device_index());

  vllm::mma_emu::mma_emu_scaled_fp8_mm_emu(
      out.data_ptr(), a.data_ptr(), b.data_ptr(),
      static_cast<const float*>(a_scales.data_ptr()),
      static_cast<const float*>(b_scales.data_ptr()),
      bias ? bias->data_ptr() : nullptr, static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)), static_cast<int>(a.size(1)),
      out_dtype == torch::headeronly::ScalarType::BFloat16
          ? ds::MmaEmuOutDtype::kBFloat16
          : ds::MmaEmuOutDtype::kFloat16,
      static_cast<int>(algorithm), static_cast<int>(f_bits),
      static_cast<int>(g_bits), static_cast<int>(group_size),
      static_cast<int>(chunk_size), stream);
}

// Exposes the accepted configuration space to the Python kernel selector, so
// the bounds and their phrasing are not restated there. Returns an empty
// string when the configuration is accepted.
std::string mma_emu_config_error(int64_t algorithm, int64_t f_bits,
                                 int64_t g_bits, int64_t group_size,
                                 int64_t chunk_size, bool is_fp4) {
  return vllm::mma_emu::mma_emu_config_error(algorithm, f_bits, g_bits,
                                             group_size, chunk_size, is_fp4);
}
