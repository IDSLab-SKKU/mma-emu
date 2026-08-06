/*
 * MMA-Emu FP8 GEMM — CUDA-core emulation kernels
 *
 * Implements FP8 E4M3 GEMM with two accumulation algorithms:
 *
 * 1. GDFS (Group-Dot-Fused-Sum): Two-level hierarchy with G-bits (group)
 *    then F-bits (fused-sum). No per-block scale factors — FP8 uses
 *    per-tensor scales applied in the epilogue.
 *
 * 2. CoFDA (Chain-of-FDA): Single-level chunked accumulation with F-bits.
 *    FDA = CoFDA with chunk_size matching the K tile dimension.
 *
 * Emulates the MMA accumulation arithmetic on CUDA cores, where every
 * alignment and truncation step is under software control. The native
 * result is produced by cutlass_scaled_mm, which is also the correctness
 * reference for these kernels.
 *
 * Emulation parameters (accepted values: design_space.cuh):
 * - f_bits: fractional bits F
 * - g_bits: GDFS intra-group bits G
 * - group_size: GDFS group size GS
 * - chunk_size: CoFDA chunk size CS
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <optional>
#include <type_traits>

#include "design_space.cuh"
#include "types.cuh"
#include "fp32_utils.cuh"
#include "tiling.cuh"
#include "accumulator.cuh"
#include "gdfs_group.cuh"
#include "fp8_e4m3.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// Tiling Configuration Alias
// ============================================================================

using FP8EmuConfig = TilingConfig<FP8EmulationTag>;

// ============================================================================
// Tiled MMA-Emu FP8 CoFDA GEMM Kernel with Shared Memory Optimization
// ============================================================================

/**
 * @brief Tiled MMA-Emu FP8 CoFDA GEMM kernel with shared memory optimization
 *
 * Block tiling (BM, BN, BK, thread count and per-thread tile) comes from
 * TilingConfig<FP8EmulationTag> in core/tiling.cuh.
 *
 * Memory layout:
 * - A: Row-major [M x K]
 * - B: Column-major [K x N] (transposed to [BN][BK] in shared memory)
 * - C: Row-major [M x N]
 *
 * @tparam CHUNK_SIZE Products per CoFDA chunk (CS)
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int CHUNK_SIZE, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_fp8_mm_cofda_kernel(
    const __nv_fp8_storage_t* __restrict__ A,  // [M, K] row-major
    const __nv_fp8_storage_t* __restrict__ B,  // [K, N] column-major
    OutDtype* __restrict__ C,                   // [M, N] row-major
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const OutDtype* __restrict__ bias,
    const int M, const int N, const int K,
    const int f_bits)
{
    // Import tiling constants
    constexpr int BM = FP8EmuConfig::BM;
    constexpr int BN = FP8EmuConfig::BN;
    constexpr int BK = FP8EmuConfig::BK;
    constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
    constexpr int BKP = BK + 1;  // padded row stride: removes stride-32 SMEM
                                 // bank conflicts.
    constexpr int A_TILE_SIZE = BM * BK;
    constexpr int B_TILE_SIZE = BN * BK;
    constexpr int A_ELEMS_PER_THREAD = (A_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD = (B_TILE_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    // Block index determines which output tile to compute
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;

    // Compute the starting position of this block's output tile
    const int block_m = block_row * BM;
    const int block_n = block_col * BN;

    // Early exit for out-of-bounds blocks
    if (block_m >= M || block_n >= N) {
        return;
    }

    // Decode-on-load packed tiles: one uint32 per operand (decode_operand +
    // pack_decoded at load). Row stride BKP (= BK+1) pads away stride-32 bank
    // conflicts. Bs is transposed: [BN][BKP].
    __shared__ uint32_t As_dec[BM * BKP];
    __shared__ uint32_t Bs_dec[BN * BKP];

    // Thread tile mapping within block
    int thread_m, thread_n;
    get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

    // Register accumulators for this thread's outputs
    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Outer loop over K dimension in chunks of BK
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ----------------------------------------
        // Load A tile [BM x BK], decode-on-load, zero-pad OOB
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE_SIZE) {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int global_row = block_m + a_row;
                int global_col = k_tile + a_col;
                const __nv_fp8_storage_t raw =
                    (global_row < M && global_col < K)
                        ? A[global_row * K + global_col]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                As_dec[a_row * BKP + a_col] = pack_decoded(d);
            }
        }

        // ----------------------------------------
        // Load B tile [BK x BN] transposed, decode-on-load, zero-pad OOB
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE_SIZE) {
                int b_row = idx / BN;  // K dimension
                int b_col = idx % BN;  // N dimension
                int global_row = k_tile + b_row;
                int global_col = block_n + b_col;
                const __nv_fp8_storage_t raw =
                    (global_row < K && global_col < N)
                        ? B[global_row + global_col * K]
                        : __nv_fp8_storage_t(0);
                const DecodedOperand d = decode_operand(static_cast<uint8_t>(raw));
                Bs_dec[b_col * BKP + b_row] = pack_decoded(d);
            }
        }
        __syncthreads();

        // ----------------------------------------
        // Compute: CoFDA chunked accumulation (one cell per thread).
        // SMEM zero-pads OOB bytes, so each chunk always has CHUNK_SIZE valid
        // (possibly zero) elements — no per-element bounds check needed.
        // ----------------------------------------
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
                        fp8_cofda_mma<CHUNK_SIZE>(a_frag, b_frag, accum[tm][tn],
                                                  f_bits);
                }
            }
        }

        __syncthreads();
    }

    // ----------------------------------------
    // Apply scales and write output
    // ----------------------------------------
    const float scale_a = *scale_a_ptr;
    const float scale_b = *scale_b_ptr;

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                // Apply scale in CUTLASS order: scale_a * (scale_b * acc)
                float val = scale_a * (scale_b * accum[tm][tn]);
                if (bias != nullptr) {
                    val += static_cast<float>(bias[out_col]);
                }
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(val);
            }
        }
    }
}

// ============================================================================
// Tiled MMA-Emu FP8 GDFS GEMM Kernel
// ============================================================================

/**
 * @brief Tiled MMA-Emu FP8 GDFS GEMM kernel.
 *
 * Two-level accumulation: group_accumulate<G, GS> → group_to_operand_fp8<F, G>
 * → gdfs_accumulate_tile<F>. No per-block scale factors (FP8 uses per-tensor
 * scales in the epilogue).
 *
 * Same tiling, shared memory, and epilogue as the CoFDA kernel.
 *
 * @tparam F Inter-group fractional bits F
 * @tparam G GDFS intra-group bits G
 * @tparam GS GDFS group size GS
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template<int GS, typename OutDtype>
__launch_bounds__(FP8EmuConfig::NUM_THREADS)
__global__ void mma_emu_scaled_fp8_mm_gdfs_kernel(
    const __nv_fp8_storage_t* __restrict__ A,  // [M, K] row-major
    const __nv_fp8_storage_t* __restrict__ B,  // [K, N] column-major
    OutDtype* __restrict__ C,                   // [M, N] row-major
    const float* __restrict__ scale_a_ptr,
    const float* __restrict__ scale_b_ptr,
    const OutDtype* __restrict__ bias,
    const int M, const int N, const int K,
    const int f_bits, const int g_bits)
{
    constexpr int BM = FP8EmuConfig::BM;
    constexpr int BN = FP8EmuConfig::BN;
    constexpr int BK = FP8EmuConfig::BK;
    constexpr int NUM_THREADS = FP8EmuConfig::NUM_THREADS;
    constexpr int THREAD_TILE_M = FP8EmuConfig::THREAD_TILE_M;
    constexpr int THREAD_TILE_N = FP8EmuConfig::THREAD_TILE_N;
    constexpr int GROUPS_PER_TILE = BK / GS;
    constexpr int BKP = BK + 1;  // padded row stride for decoded tiles: removes
                                 // stride bank conflicts.
    constexpr int A_TILE = BM * BK;   // 1 byte/elem -> 1 word/elem (no nibble pack)
    constexpr int B_TILE = BN * BK;
    constexpr int A_ELEMS_PER_THREAD = (A_TILE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int B_ELEMS_PER_THREAD = (B_TILE + NUM_THREADS - 1) / NUM_THREADS;

    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;

    const int block_m = block_row * BM;
    const int block_n = block_col * BN;

    if (block_m >= M || block_n >= N) {
        return;
    }

    __shared__ uint32_t As_dec[BM * BKP];  // decode-on-load: one word per FP8 elem
    __shared__ uint32_t Bs_dec[BN * BKP];

    int thread_m, thread_n;
    get_thread_tile_position<FP8EmuConfig>(tid, thread_m, thread_n);

    float accum[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; j++) {
            accum[i][j] = 0.0f;
        }
    }

    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ----------------------------------------
        // Load A tile [BM, BK]: decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < A_TILE) {
                int a_row = idx / BK;
                int a_col = idx % BK;
                int global_row = block_m + a_row;
                int global_col = k_tile + a_col;
                uint8_t byte = (global_row < M && global_col < K)
                                   ? static_cast<uint8_t>(A[global_row * K + global_col])
                                   : uint8_t(0);
                As_dec[a_row * BKP + a_col] = pack_decoded(decode_operand(byte));
            }
        }

        // ----------------------------------------
        // Load B tile [BN, BK] (transposed): decode-on-load, zero-pad
        // ----------------------------------------
        #pragma unroll
        for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
            int idx = tid + i * NUM_THREADS;
            if (idx < B_TILE) {
                int b_row = idx / BN;
                int b_col = idx % BN;
                int global_row = k_tile + b_row;
                int global_col = block_n + b_col;
                uint8_t byte = (global_row < K && global_col < N)
                                   ? static_cast<uint8_t>(B[global_row + global_col * K])
                                   : uint8_t(0);
                Bs_dec[b_col * BKP + b_row] = pack_decoded(decode_operand(byte));
            }
        }
        __syncthreads();

        // ----------------------------------------
        // GDFS compute (zero-padded SMEM -> no per-element bounds check)
        // ----------------------------------------
        #pragma unroll
        for (int tm = 0; tm < THREAD_TILE_M; tm++) {
            #pragma unroll
            for (int tn = 0; tn < THREAD_TILE_N; tn++) {
                Operand tile_groups[GROUPS_PER_TILE];
                #pragma unroll
                for (int g = 0; g < GROUPS_PER_TILE; g++) {
                    const DecodedFrag a_frag{&As_dec[(thread_m + tm) * BKP + g * GS]};
                    const DecodedFrag b_frag{&Bs_dec[(thread_n + tn) * BKP + g * GS]};
                    GroupResult gr =
                        fp8_gdfs_group_accumulate_predecoded<GS>(a_frag, b_frag,
                                                                 g_bits);
                    if (gr.all_zero) {
                        tile_groups[g] = make_zero_operand();
                    } else {
                        tile_groups[g] = group_to_operand_fp8(
                            gr.mantissa_sum, gr.max_exp, f_bits, g_bits);
                    }
                }
                accum[tm][tn] = gdfs_accumulate_tile<GROUPS_PER_TILE>(
                    tile_groups, accum[tm][tn], f_bits);
            }
        }

        __syncthreads();
    }

    // ----------------------------------------
    // Apply per-tensor scales and write output
    // ----------------------------------------
    const float scale_a = *scale_a_ptr;
    const float scale_b = *scale_b_ptr;

    #pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
        #pragma unroll
        for (int tn = 0; tn < THREAD_TILE_N; tn++) {
            int out_row = block_m + thread_m + tm;
            int out_col = block_n + thread_n + tn;

            if (out_row < M && out_col < N) {
                float val = scale_a * (scale_b * accum[tm][tn]);
                if (bias != nullptr) {
                    val += static_cast<float>(bias[out_col]);
                }
                C[out_row * N + out_col] = float_to_output_rn<OutDtype>(val);
            }
        }
    }
}

// ============================================================================
// Kernel Dispatch
// ============================================================================
//
// Only CHUNK_SIZE, GS and the output dtype select a template instantiation.
// F and G are kernel arguments, so one instantiation per (CS or GS) x dtype
// pair serves their whole accepted range.

namespace detail {

template <typename OutDtype>
inline void launch_fp8_cofda(
    dim3 grid, dim3 block, cudaStream_t stream,
    const __nv_fp8_storage_t* a, const __nv_fp8_storage_t* b, OutDtype* c,
    const float* scale_a, const float* scale_b, const OutDtype* bias,
    int M, int N, int K, int f_bits, int chunk_size) {
  switch (chunk_size) {
    case 16:
      mma_emu_scaled_fp8_mm_cofda_kernel<16, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, bias, M, N, K, f_bits);
      break;
    case 32:
      mma_emu_scaled_fp8_mm_cofda_kernel<32, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, bias, M, N, K, f_bits);
      break;
    default:
      break;  // rejected by the operator entry point
  }
}

template <typename OutDtype>
inline void launch_fp8_gdfs(
    dim3 grid, dim3 block, cudaStream_t stream,
    const __nv_fp8_storage_t* a, const __nv_fp8_storage_t* b, OutDtype* c,
    const float* scale_a, const float* scale_b, const OutDtype* bias,
    int M, int N, int K, int f_bits, int g_bits, int group_size) {
  switch (group_size) {
    case 8:
      mma_emu_scaled_fp8_mm_gdfs_kernel<8, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, bias, M, N, K, f_bits, g_bits);
      break;
    case 16:
      mma_emu_scaled_fp8_mm_gdfs_kernel<16, OutDtype><<<grid, block, 0, stream>>>(
          a, b, c, scale_a, scale_b, bias, M, N, K, f_bits, g_bits);
      break;
    default:
      break;  // rejected by the operator entry point
  }
}

template <typename OutDtype>
inline void dispatch_fp8(
    dim3 grid, dim3 block, cudaStream_t stream, const void* a, const void* b,
    void* out, const float* scale_a, const float* scale_b, const void* bias,
    int M, int N, int K, int algorithm, int f_bits, int g_bits,
    int group_size, int chunk_size) {
  const auto* a_ptr = static_cast<const __nv_fp8_storage_t*>(a);
  const auto* b_ptr = static_cast<const __nv_fp8_storage_t*>(b);
  auto* c_ptr = static_cast<OutDtype*>(out);
  const auto* bias_ptr = static_cast<const OutDtype*>(bias);

  if (algorithm == design_space::kGDFS) {
    launch_fp8_gdfs<OutDtype>(grid, block, stream, a_ptr, b_ptr, c_ptr, scale_a,
                              scale_b, bias_ptr, M, N, K, f_bits, g_bits,
                              group_size);
  } else {
    launch_fp8_cofda<OutDtype>(grid, block, stream, a_ptr, b_ptr, c_ptr, scale_a,
                               scale_b, bias_ptr, M, N, K, f_bits, chunk_size);
  }
}

}  // namespace detail

// ============================================================================
// Host Entry
// ============================================================================
//
// Takes raw pointers rather than tensors: the torch boundary lives entirely in
// mma_emu_scaled_fp8_mm_entry.cu, which validates every argument before
// calling here.

inline void mma_emu_scaled_fp8_mm_emu(
    void* out, const void* a, const void* b, const float* a_scales,
    const float* b_scales, const void* bias, int M, int N, int K,
    design_space::MmaEmuOutDtype out_dtype, int algorithm, int f_bits, int g_bits,
    int group_size, int chunk_size, cudaStream_t stream) {
  using Config = FP8EmuConfig;
  dim3 grid((N + Config::BN - 1) / Config::BN,
            (M + Config::BM - 1) / Config::BM);
  dim3 block(Config::NUM_THREADS);

  if (out_dtype == design_space::MmaEmuOutDtype::kBFloat16) {
    detail::dispatch_fp8<__nv_bfloat16>(grid, block, stream, a, b, out, a_scales,
                                        b_scales, bias, M, N, K, algorithm,
                                        f_bits, g_bits, group_size, chunk_size);
  } else {
    detail::dispatch_fp8<__half>(grid, block, stream, a, b, out, a_scales,
                                 b_scales, bias, M, N, K, algorithm, f_bits,
                                 g_bits, group_size, chunk_size);
  }
}

}  // namespace mma_emu
}  // namespace vllm
