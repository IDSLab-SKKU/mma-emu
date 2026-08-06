#include <cuda_runtime.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>

#include "libtorch_stable/torch_utils.h"

#include "mma_emu_config_check.h"
#include "mma_emu_moe_grouped_mm_kernels.cuh"

namespace ds = vllm::mma_emu::design_space;

// Emulated FP8 grouped MoE GEMM. Mirrors the cutlass_moe_mm interface so the
// two are interchangeable, with the accumulation configuration appended.
//
// CoFDA only. GDFS is available for dense layers but not for MoE, so the
// operator takes no G or group size and rejects any other algorithm rather
// than quietly computing something else.
//
// a_strides, b_strides and c_strides are accepted for parity with
// cutlass_moe_mm and are unused: this kernel derives its addressing from
// expert_offsets and the problem sizes.
void mma_emu_moe_mm(torch::stable::Tensor& out_tensors,
                    const torch::stable::Tensor& a_tensors,
                    const torch::stable::Tensor& b_tensors,
                    const torch::stable::Tensor& a_scales,
                    const torch::stable::Tensor& b_scales,
                    const torch::stable::Tensor& expert_offsets,
                    const torch::stable::Tensor& problem_sizes,
                    const torch::stable::Tensor& a_strides,
                    const torch::stable::Tensor& b_strides,
                    const torch::stable::Tensor& c_strides, bool per_act_token,
                    bool per_out_ch, int64_t algorithm, int64_t f_bits,
                    int64_t chunk_size) {
  (void)a_strides;
  (void)b_strides;
  (void)c_strides;

  STD_TORCH_CHECK(
      a_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
          b_tensors.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn,
      "MMA-Emu moe_mm: a and b must be float8_e4m3fn");

  STD_TORCH_CHECK(a_tensors.dim() == 2 && b_tensors.dim() == 3 &&
                      out_tensors.dim() == 2,
                  "MMA-Emu moe_mm: expected a [P, K], b [E, N, K], out [P, N]");

  STD_TORCH_CHECK(a_tensors.stride(1) == 1 && out_tensors.stride(1) == 1,
                  "MMA-Emu moe_mm: a and out must be row-major");
  STD_TORCH_CHECK(b_tensors.stride(2) == 1,
                  "MMA-Emu moe_mm: b must be [E, N, K] with contiguous K");

  // The emulation applies one scale per expert in its epilogue, so a per-token
  // or per-output-channel scale vector has nowhere to go.
  STD_TORCH_CHECK(!per_act_token && !per_out_ch,
                  "MMA-Emu moe_mm: only per-tensor scales are supported");
  STD_TORCH_CHECK(a_scales.numel() == 1,
                  "MMA-Emu moe_mm: a_scales must be a scalar");
  STD_TORCH_CHECK(
      a_scales.scalar_type() == torch::headeronly::ScalarType::Float &&
          b_scales.scalar_type() == torch::headeronly::ScalarType::Float,
      "MMA-Emu moe_mm: scales must be float32");

  STD_TORCH_CHECK(
      expert_offsets.scalar_type() == torch::headeronly::ScalarType::Long,
      "MMA-Emu moe_mm: expert_offsets must be int64");
  STD_TORCH_CHECK(
      problem_sizes.scalar_type() == torch::headeronly::ScalarType::Int,
      "MMA-Emu moe_mm: problem_sizes must be int32");

  const int64_t num_experts = b_tensors.size(0);
  const int64_t n = b_tensors.size(1);
  const int64_t k = b_tensors.size(2);
  const int64_t total_rows = a_tensors.size(0);

  STD_TORCH_CHECK(a_tensors.size(1) == k,
                  "MMA-Emu moe_mm: a and b disagree on K");
  STD_TORCH_CHECK(out_tensors.size(0) == total_rows && out_tensors.size(1) == n,
                  "MMA-Emu moe_mm: out must be [", total_rows, ", ", n, "]");
  STD_TORCH_CHECK(b_scales.numel() == num_experts,
                  "MMA-Emu moe_mm: b_scales must have one entry per expert");
  STD_TORCH_CHECK(expert_offsets.numel() >= num_experts,
                  "MMA-Emu moe_mm: expert_offsets is shorter than the expert count");
  STD_TORCH_CHECK(problem_sizes.numel() >= num_experts * 3,
                  "MMA-Emu moe_mm: problem_sizes must be [E, 3]");

  const auto out_dtype = out_tensors.scalar_type();
  STD_TORCH_CHECK(out_dtype == torch::headeronly::ScalarType::BFloat16 ||
                      out_dtype == torch::headeronly::ScalarType::Half,
                  "MMA-Emu moe_mm: out must be BFloat16 or Half");

  STD_TORCH_CHECK(algorithm == ds::kCoFDA,
                  "MMA-Emu moe_mm: only algorithm ", int64_t{ds::kCoFDA},
                  " (CoFDA) is supported for MoE, got ", algorithm);

  // Backstop: the Python selector rejects a bad configuration earlier and with
  // the same wording, but this operator is reachable directly. G and the group
  // size are GDFS-only, so the values passed here are never consulted.
  const std::string err = vllm::mma_emu::mma_emu_config_error(
      algorithm, f_bits, /*g_bits=*/0, /*group_size=*/0, chunk_size,
      /*is_fp4=*/false);
  STD_TORCH_CHECK(err.empty(), "MMA-Emu moe_mm: ", err);

  const torch::stable::accelerator::DeviceGuard device_guard(
      a_tensors.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(a_tensors.get_device_index());

  vllm::mma_emu::mma_emu_moe_grouped_mm_emu(
      out_tensors.data_ptr(), a_tensors.data_ptr(), b_tensors.data_ptr(),
      static_cast<const float*>(a_scales.data_ptr()),
      static_cast<const float*>(b_scales.data_ptr()),
      static_cast<const int32_t*>(problem_sizes.data_ptr()),
      static_cast<const int64_t*>(expert_offsets.data_ptr()),
      static_cast<int>(num_experts), static_cast<int>(total_rows),
      static_cast<int>(n), static_cast<int>(k),
      out_dtype == torch::headeronly::ScalarType::BFloat16
          ? ds::MmaEmuOutDtype::kBFloat16
          : ds::MmaEmuOutDtype::kFloat16,
      static_cast<int>(f_bits), static_cast<int>(chunk_size), stream);
}
