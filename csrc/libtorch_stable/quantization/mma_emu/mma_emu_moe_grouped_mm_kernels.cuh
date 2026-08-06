// Grouped FP8 MoE GEMM under emulated CoFDA accumulation.
//
// CoFDA only. GDFS is supported for dense layers but not here.
//
// Single launch over all experts. The per-tile arithmetic is byte-for-byte the
// dense kernel in mma_emu_scaled_fp8_mm_kernels.cuh; the only additions are an
// expert dimension and per-expert base pointers and weight scale.
//
// Nothing here is architecture-specific — it runs on CUDA cores wherever the
// dense kernel does — so the name carries no SM suffix, unlike the original.
//
// Grid: blockIdx.z = expert e, blockIdx.y = M tile, blockIdx.x = N tile.
//   Launch grid.z == E (number of experts); each block reads its own
//   problem_sizes[e]/expert_offsets[e]. Tiling comes from FP8EmuConfig.
// Layout (identical to cutlass_moe_mm):
//   A: [P, K] row-major, all experts concatenated; expert e starts at
//      row expert_offsets[e].
//   B: [E, N, K]; each expert's weight is [K, N] col-major (B[e] base = e*N*K,
//      element (n,k) at k + n*K).
//   C: [P, N] row-major, same row offsets as A.
//   scale_a: scalar, shared across experts. scale_b: [E], per-expert.
//   problem_sizes: [E,3] = (M_e, N, K); only the M_e column is read here
//      (N and K are the same for every expert and are passed as kernel args).
#pragma once

#include "design_space.cuh"
#include "mma_emu_scaled_fp8_mm_kernels.cuh"

namespace vllm {
namespace mma_emu {

template <int CHUNK_SIZE, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_moe_grouped_cofda_kernel(
    const __nv_fp8_storage_t* __restrict__ A,    // [P, K] row-major (all experts)
    const __nv_fp8_storage_t* __restrict__ B,    // [E, N, K]; each expert [K,N] col-major
    OutDtype* __restrict__ C,                     // [P, N] row-major
    const float* __restrict__ scale_a_ptr,        // scalar, shared across experts
    const float* __restrict__ scale_b_ptr,        // [E] per-expert
    const int32_t* __restrict__ problem_sizes,    // [E,3] = (M_e, N, K)
    const int64_t* __restrict__ expert_offsets,   // [E] row start in A/C
    const int N, const int K, const int f_bits) {
  constexpr int BM = FP8EmuConfig::BM;
  constexpr int BN = FP8EmuConfig::BN;
  constexpr int BK = FP8EmuConfig::BK;
  constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
  constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
  constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
  constexpr int BKP = BK + 1;
  constexpr int A_TILE_SIZE = BM * BK;
  constexpr int B_TILE_SIZE = BN * BK;
  constexpr int A_ELEMS_PER_THREAD = (A_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;
  constexpr int B_ELEMS_PER_THREAD = (B_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;

  const int e = blockIdx.z;
  const int M = problem_sizes[e * 3 + 0];  // M_e (rows for this expert)
  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;
  if (block_m >= M || block_n >= N) return;  // ragged-M / empty-expert early exit

  const int64_t row0 = expert_offsets[e];
  const __nv_fp8_storage_t* Ae = A + row0 * static_cast<int64_t>(K);
  const __nv_fp8_storage_t* Be = B + static_cast<int64_t>(e) * N * K;
  OutDtype* Ce = C + row0 * static_cast<int64_t>(N);
  const float scale_a = *scale_a_ptr;
  const float scale_b = scale_b_ptr[e];

  const int tid = threadIdx.x;
  __shared__ uint32_t As_dec[BM * BKP];
  __shared__ uint32_t Bs_dec[BN * BKP];

  int thread_m, thread_n;
  get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

  float accum[THREAD_TILE_M][THREAD_TILE_N];
  #pragma unroll
  for (int i = 0; i < THREAD_TILE_M; i++)
    #pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j++) accum[i][j] = 0.0f;

  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    #pragma unroll
    for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
      int idx = tid + i * NUM_THREADS;
      if (idx < A_TILE_SIZE) {
        int a_row = idx / BK, a_col = idx % BK;
        int gr = block_m + a_row, gc = k_tile + a_col;
        const __nv_fp8_storage_t raw =
            (gr < M && gc < K) ? Ae[gr * static_cast<int64_t>(K) + gc]
                               : __nv_fp8_storage_t(0);
        As_dec[a_row * BKP + a_col] =
            pack_decoded(decode_operand(static_cast<uint8_t>(raw)));
      }
    }
    #pragma unroll
    for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
      int idx = tid + i * NUM_THREADS;
      if (idx < B_TILE_SIZE) {
        int b_row = idx / BN, b_col = idx % BN;  // b_row=K, b_col=N
        int gr = k_tile + b_row, gc = block_n + b_col;
        const __nv_fp8_storage_t raw =
            (gr < K && gc < N) ? Be[gr + static_cast<int64_t>(gc) * K]
                               : __nv_fp8_storage_t(0);
        Bs_dec[b_col * BKP + b_row] =
            pack_decoded(decode_operand(static_cast<uint8_t>(raw)));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
      #pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        const int a_off = (thread_m + tm) * BKP;
        const int b_off = (thread_n + tn) * BKP;
        #pragma unroll
        for (int k_start = 0; k_start < BK; k_start += CHUNK_SIZE) {
          const DecodedFrag a_frag{&As_dec[a_off + k_start]};
          const DecodedFrag b_frag{&Bs_dec[b_off + k_start]};
          accum[tm][tn] =
              fp8_cofda_mma<CHUNK_SIZE>(a_frag, b_frag, accum[tm][tn], f_bits);
        }
      }
    }
    __syncthreads();
  }

  #pragma unroll
  for (int tm = 0; tm < THREAD_TILE_M; tm++) {
    #pragma unroll
    for (int tn = 0; tn < THREAD_TILE_N; tn++) {
      int out_row = block_m + thread_m + tm;
      int out_col = block_n + thread_n + tn;
      if (out_row < M && out_col < N) {
        float val = scale_a * (scale_b * accum[tm][tn]);
        Ce[out_row * static_cast<int64_t>(N) + out_col] =
            float_to_output_rn<OutDtype>(val);
      }
    }
  }
}

// ============================================================================
// Kernel Dispatch
// ============================================================================
//
// As in the dense case, only CHUNK_SIZE and the output dtype select a
// template instantiation; F is a kernel argument. The kernel this was ported
// from accepted CS = 32 only, because its dispatch enumerated F; that
// restriction is gone.

namespace detail {

template <typename OutDtype>
inline void launch_moe_cofda(
    dim3 grid, dim3 block, cudaStream_t stream,
    const __nv_fp8_storage_t* a, const __nv_fp8_storage_t* b, OutDtype* c,
    const float* scale_a, const float* scale_b, const int32_t* problem_sizes,
    const int64_t* expert_offsets, int N, int K, int f_bits, int chunk_size) {
  switch (chunk_size) {
    case 16:
      mma_emu_moe_grouped_cofda_kernel<16, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, problem_sizes, expert_offsets, N, K, f_bits);
      break;
    case 32:
      mma_emu_moe_grouped_cofda_kernel<32, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, problem_sizes, expert_offsets, N, K, f_bits);
      break;
    default:
      break;  // rejected by the operator entry point
  }
}

template <typename OutDtype>
inline void dispatch_moe(
    dim3 grid, dim3 block, cudaStream_t stream, const void* a, const void* b,
    void* out, const float* scale_a, const float* scale_b,
    const int32_t* problem_sizes, const int64_t* expert_offsets, int N, int K,
    int f_bits, int chunk_size) {
  launch_moe_cofda<OutDtype>(
      grid, block, stream, static_cast<const __nv_fp8_storage_t*>(a),
      static_cast<const __nv_fp8_storage_t*>(b), static_cast<OutDtype*>(out),
      scale_a, scale_b, problem_sizes, expert_offsets, N, K, f_bits,
      chunk_size);
}

}  // namespace detail

// ============================================================================
// Host Entry
// ============================================================================
//
// Takes raw pointers rather than tensors: the torch boundary lives entirely in
// mma_emu_moe_grouped_mm_entry.cu, which validates every argument before
// calling here.
//
// total_rows is the summed M over all experts, and is used to size grid.y. It
// over-covers, since it bounds any single expert's M_e; blocks past an
// expert's rows exit immediately. A scheduler that packs the ragged M
// dimension would launch fewer of them.

inline void mma_emu_moe_grouped_mm_emu(
    void* out, const void* a, const void* b, const float* a_scales,
    const float* b_scales, const int32_t* problem_sizes,
    const int64_t* expert_offsets, int num_experts, int total_rows, int N,
    int K, design_space::MmaEmuOutDtype out_dtype, int f_bits, int chunk_size,
    cudaStream_t stream) {
  using Config = FP8EmuConfig;
  dim3 grid((N + Config::BN - 1) / Config::BN,
            (total_rows + Config::BM - 1) / Config::BM, num_experts);
  dim3 block(Config::NUM_THREADS);

  if (out_dtype == design_space::MmaEmuOutDtype::kBFloat16) {
    detail::dispatch_moe<__nv_bfloat16>(
        grid, block, stream, a, b, out, a_scales, b_scales, problem_sizes,
        expert_offsets, N, K, f_bits, chunk_size);
  } else {
    detail::dispatch_moe<__half>(
        grid, block, stream, a, b, out, a_scales, b_scales, problem_sizes,
        expert_offsets, N, K, f_bits, chunk_size);
  }
}

}  // namespace mma_emu
}  // namespace vllm
