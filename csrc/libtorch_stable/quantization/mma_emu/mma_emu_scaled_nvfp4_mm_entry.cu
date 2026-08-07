#include <cuda_runtime.h>

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>

#include "libtorch_stable/torch_utils.h"

#include "mma_emu_config_check.h"
#include "mma_emu_scaled_nvfp4_mm_kernels.cuh"

namespace ds = vllm::mma_emu::design_space;

namespace {

// FP4 is packed two elements per byte.
constexpr auto kFloat4E2m1x2 = torch::headeronly::ScalarType::Byte;
// NVFP4 block scales are UE4M3.
constexpr auto kScaleDtype = torch::headeronly::ScalarType::Float8_e4m3fn;

constexpr int64_t round_up(int64_t x, int64_t y) { return (x + y - 1) / y * y; }

}  // namespace

// Emulated NVFP4 scaled mm. Mirrors the cutlass_scaled_fp4_mm interface, with
// the accumulation configuration appended.
void mma_emu_scaled_nvfp4_mm(
    torch::stable::Tensor& out, const torch::stable::Tensor& a,
    const torch::stable::Tensor& b, const torch::stable::Tensor& a_sf,
    const torch::stable::Tensor& b_sf, const torch::stable::Tensor& alpha,
    int64_t algorithm, int64_t f_bits, int64_t g_bits) {
  STD_TORCH_CHECK(a.scalar_type() == kFloat4E2m1x2 &&
                  b.scalar_type() == kFloat4E2m1x2);
  STD_TORCH_CHECK(a_sf.scalar_type() == kScaleDtype &&
                  b_sf.scalar_type() == kScaleDtype);
  STD_TORCH_CHECK(alpha.scalar_type() == torch::headeronly::ScalarType::Float &&
                  alpha.numel() == 1);
  STD_TORCH_CHECK(a.is_contiguous() && b.is_contiguous() &&
                  a_sf.is_contiguous() && b_sf.is_contiguous() &&
                  out.is_contiguous());

  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && out.dim() == 2 &&
                  a_sf.dim() == 2 && b_sf.dim() == 2);

  // a is [M, K/2] packed, b is [N, K/2] packed, out is [M, N].
  const int64_t m = a.size(0);
  const int64_t n = b.size(0);
  const int64_t k = a.size(1) * 2;

  STD_TORCH_CHECK(
      a.size(1) == b.size(1),
      "MMA-Emu scaled_nvfp4_mm: a and b shapes cannot be multiplied");
  STD_TORCH_CHECK(out.size(0) == m && out.size(1) == n);

  // K must be divisible by 32 (block of 16, packed two per byte) and N by 32
  // for the tile loads.
  constexpr int64_t kAlignment = 32;
  STD_TORCH_CHECK(k % kAlignment == 0,
                  "MMA-Emu scaled_nvfp4_mm: K must be divisible by ",
                  kAlignment, ", got ", k);
  STD_TORCH_CHECK(n % kAlignment == 0,
                  "MMA-Emu scaled_nvfp4_mm: N must be divisible by ",
                  kAlignment, ", got ", n);

  // Block scales arrive padded and swizzled.
  const int64_t rounded_m = round_up(m, 128);
  const int64_t rounded_n = round_up(n, 128);
  const int64_t rounded_k = round_up(k / 16, 4);
  STD_TORCH_CHECK(
      a_sf.size(0) == rounded_m && a_sf.size(1) == rounded_k,
      "MMA-Emu scaled_nvfp4_mm: a_sf must be padded and swizzled to (",
      rounded_m, ", ", rounded_k, ")");
  STD_TORCH_CHECK(
      b_sf.size(0) == rounded_n && b_sf.size(1) == rounded_k,
      "MMA-Emu scaled_nvfp4_mm: b_sf must be padded and swizzled to (",
      rounded_n, ", ", rounded_k, ")");

  const auto out_dtype = out.scalar_type();
  STD_TORCH_CHECK(out_dtype == torch::headeronly::ScalarType::BFloat16 ||
                      out_dtype == torch::headeronly::ScalarType::Half,
                  "MMA-Emu scaled_nvfp4_mm: out must be BFloat16 or Half");

  // Backstop: the Python kernel selector rejects a bad configuration earlier
  // and with the same wording, but this operator is reachable directly.
  // CS and GS are fixed at 16 in the NVFP4 kernels, so they are not arguments.
  const std::string err = vllm::mma_emu::mma_emu_config_error(
      algorithm, f_bits, g_bits, /*group_size=*/16, /*chunk_size=*/16,
      /*is_fp4=*/true);
  STD_TORCH_CHECK(err.empty(), "MMA-Emu scaled_nvfp4_mm: ", err);

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(a.get_device_index());

  vllm::mma_emu::mma_emu_scaled_nvfp4_mm_emu(
      out.data_ptr(), a.data_ptr(), b.data_ptr(), a_sf.data_ptr(),
      b_sf.data_ptr(), static_cast<const float*>(alpha.data_ptr()),
      static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
      out_dtype == torch::headeronly::ScalarType::BFloat16
          ? ds::MmaEmuOutDtype::kBFloat16
          : ds::MmaEmuOutDtype::kFloat16,
      static_cast<int>(algorithm), static_cast<int>(f_bits),
      static_cast<int>(g_bits), stream);
}
