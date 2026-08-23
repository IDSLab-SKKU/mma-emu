/*
 * MMA-Emu NVFP4 GEMM — CUDA-core emulation kernels
 *
 * Implements NVFP4 (E2M1) GEMM with two accumulation algorithms:
 * 1. GDFS (Group-Dot-Fused-Sum): group-level accumulation with configurable
 *    G and F from core/design_space.cuh.
 * 2. CoFDA (Chunked FDA): product-level scale application with chunked
 *    accumulation, CS fixed at 16; F from core/design_space.cuh.
 *
 * Emulates the accumulation arithmetic of the Blackwell OMMA.SF (NVFP4)
 * instruction on CUDA cores. The native result is produced by
 * cutlass_scaled_fp4_mm.
 *
 * Key Features:
 * - Packed FP4 input (2 values per byte)
 * - UE4M3 block scale factors (block size = 16)
 * - Alpha epilogue for global scale application
 *
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "design_space.cuh"
#include "mma_emu_instantiation.cuh"
#include "types.cuh"
#include "fp32_utils.cuh"
#include "tiling.cuh"
#include "accumulator.cuh"
#include "gdfs_group.cuh"
#include "fp4_e2m1.cuh"
#include "scale_swizzle.cuh"
#include "nvfp4_ue4m3.cuh"

namespace vllm {
namespace mma_emu {

// ============================================================================
// Tiling Configuration Alias
// ============================================================================

using NVFP4EmuConfig = TilingConfig<NVFP4EmulationTag>;

// ============================================================================
// Tiled MMA-Emu NVFP4 GDFS GEMM Kernel
// ============================================================================

/**
 * @brief Tiled MMA-Emu NVFP4 GDFS GEMM kernel with shared memory optimization.
 *
 * Block tiling (BM, BN, BK, thread count and per-thread tile) comes from
 * TilingConfig<NVFP4EmulationTag> in core/tiling.cuh. Shared memory
 * for efficient memory access patterns. Each thread computes a 2x1 output tile.
 *
 * Key optimization: Fused GDFS accumulation per K-tile eliminates the need for
 * large local arrays. Each K-tile's groups (BK / GS of them) are fused with
 * the running FP32 accumulator following the FDA pattern.
 *
 * Memory hierarchy:
 * - Global Memory: A[M, K/2], B[N, K/2], A_sf[swizzled], B_sf[swizzled]
 * - Shared Memory: As[BM, BK/2], Bs[BN, BK/2], A_sf_s[BM, 4], B_sf_s[BN, 4]
 * - Registers: accum[THREAD_TILE_M][THREAD_TILE_N],
 * tile_groups[GROUPS_PER_TILE]
 *
 * @tparam F Inter-group fractional bits F
 * @tparam G GDFS intra-group bits G
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template <int F, int G, int GS, typename OutDtype>
__launch_bounds__(NVFP4EmuConfig::NUM_THREADS) __global__
    void mma_emu_scaled_nvfp4_mm_emu_kernel(
        const uint8_t* __restrict__ A,     // [M, K/2] packed FP4
        const uint8_t* __restrict__ B,     // [N, K/2] packed FP4
        OutDtype* __restrict__ C,          // [M, N] output
        const uint8_t* __restrict__ A_sf,  // Swizzled block scales for A
        const uint8_t* __restrict__ B_sf,  // Swizzled block scales for B
        const float* __restrict__ alpha,   // [1] global scale
        const int M, const int N, const int K, const int f_bits,
        const int g_bits) {
  // Literals fold through the __forceinline__ arithmetic; the sentinels keep
  // the argument path for anything not enumerated.
  const int f = (F == kRuntimeF) ? f_bits : F;
  const int g = (G == kRuntimeG) ? g_bits : G;

  // Import tiling constants
  constexpr int BM = NVFP4EmuConfig::BM;
  constexpr int BN = NVFP4EmuConfig::BN;
  constexpr int BK = NVFP4EmuConfig::BK;
  constexpr int BK_PACKED = NVFP4EmuConfig::BK_PACKED;
  constexpr int NUM_THREADS = NVFP4EmuConfig::NUM_THREADS;
  constexpr int THREAD_TILE_M = NVFP4EmuConfig::THREAD_TILE_M;
  constexpr int THREAD_TILE_N = NVFP4EmuConfig::THREAD_TILE_N;
  constexpr int BLOCK_SIZE = NVFP4EmuConfig::BLOCK_SIZE;
  constexpr int SF_PER_TILE = NVFP4EmuConfig::SF_PER_TILE;

  // Derived from template parameter GS
  constexpr int GROUPS_PER_TILE = BK / GS;
  constexpr int GROUPS_PER_SCALE = BLOCK_SIZE / GS;

  constexpr int BKP = BK + 1;  // padded row stride for decoded tiles: removes
                               // stride bank conflicts.
  constexpr int A_TILE_PACKED = BM * BK_PACKED;  // packed-byte tile size
  constexpr int B_TILE_PACKED = BN * BK_PACKED;
  constexpr int A_ELEMS_PER_THREAD =
      (A_TILE_PACKED + NUM_THREADS - 1) / NUM_THREADS;
  constexpr int B_ELEMS_PER_THREAD =
      (B_TILE_PACKED + NUM_THREADS - 1) / NUM_THREADS;

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

  // ========================================
  // Shared Memory Declaration
  // ========================================
  // Decode-on-load: one packed uint32 per FP4 elem, row stride BKP (= BK+1)
  // pads away stride bank conflicts. Scale tiles stay uint8 (de-swizzled).
  __shared__ uint32_t As_dec[BM * BKP];
  __shared__ uint32_t Bs_dec[BN * BKP];
  __shared__ uint8_t A_sf_s[NVFP4EmuConfig::SMEM_A_SF_SIZE];
  __shared__ uint8_t B_sf_s[NVFP4EmuConfig::SMEM_B_SF_SIZE];

  // Thread tile mapping within block
  int thread_m, thread_n;
  get_thread_tile_position<NVFP4EmuConfig>(tid, thread_m, thread_n);

  // ========================================
  // Register Accumulators (FP32)
  // ========================================
  float accum[THREAD_TILE_M][THREAD_TILE_N];
#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j++) {
      accum[i][j] = 0.0f;
    }
  }

  // Precompute constants
  const int K_packed = K / 2;
  const int num_k_tiles = (K + BK - 1) / BK;

  // ========================================
  // OUTER LOOP: K-dimension tiles
  // ========================================
  for (int kt = 0; kt < num_k_tiles; kt++) {
    const int k_tile_start = kt * BK;
    const int k_tile_packed_start = k_tile_start / 2;
    const int k_block_start = kt * SF_PER_TILE;

// ----------------------------------------
// Load A tile [BM, BK]: unpack 2 nibbles/byte, decode-on-load, zero-pad
// ----------------------------------------
#pragma unroll
    for (int i = 0; i < A_ELEMS_PER_THREAD; i++) {
      int idx = tid + i * NUM_THREADS;
      if (idx < A_TILE_PACKED) {
        int a_row = idx / BK_PACKED;
        int a_col = idx % BK_PACKED;  // packed byte col [0, BK_PACKED)
        int global_row = block_m + a_row;
        int global_col = k_tile_packed_start + a_col;
        uint8_t byte = (global_row < M && global_col < K_packed)
                           ? A[global_row * K_packed + global_col]
                           : static_cast<uint8_t>(__nv_fp4_storage_t(0));
        // low nibble -> k=2*a_col (even), high nibble -> k=2*a_col+1 (odd)
        As_dec[a_row * BKP + 2 * a_col] =
            pack_decoded_fp4(decode_fp4_operand(byte & 0x0F));
        As_dec[a_row * BKP + 2 * a_col + 1] =
            pack_decoded_fp4(decode_fp4_operand(byte >> 4));
      }
    }

// ----------------------------------------
// Load B tile [BN, BK]: unpack 2 nibbles/byte, decode-on-load, zero-pad
// ----------------------------------------
#pragma unroll
    for (int i = 0; i < B_ELEMS_PER_THREAD; i++) {
      int idx = tid + i * NUM_THREADS;
      if (idx < B_TILE_PACKED) {
        int b_row = idx / BK_PACKED;
        int b_col = idx % BK_PACKED;
        int global_row = block_n + b_row;
        int global_col = k_tile_packed_start + b_col;
        uint8_t byte = (global_row < N && global_col < K_packed)
                           ? B[global_row * K_packed + global_col]
                           : static_cast<uint8_t>(__nv_fp4_storage_t(0));
        Bs_dec[b_row * BKP + 2 * b_col] =
            pack_decoded_fp4(decode_fp4_operand(byte & 0x0F));
        Bs_dec[b_row * BKP + 2 * b_col + 1] =
            pack_decoded_fp4(decode_fp4_operand(byte >> 4));
      }
    }

    // ----------------------------------------
    // Load A scale factors (de-swizzle during load)
    // ----------------------------------------
    {
      constexpr int A_sf_tile_size = NVFP4EmuConfig::SMEM_A_SF_SIZE;
      const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

      for (int idx = tid; idx < A_sf_tile_size; idx += NUM_THREADS) {
        int m_local = idx / SF_PER_TILE;
        int sf_local = idx % SF_PER_TILE;
        int m_global = block_m + m_local;
        int k_block_global = k_block_start + sf_local;

        if (m_global < M && k_block_global < num_k_blocks) {
          A_sf_s[idx] =
              swizzle::read_scale(A_sf, m_global, k_block_global, M, K);
        } else {
          A_sf_s[idx] = 0;
        }
      }
    }

    // ----------------------------------------
    // Load B scale factors (de-swizzle during load)
    // ----------------------------------------
    {
      constexpr int B_sf_tile_size = NVFP4EmuConfig::SMEM_B_SF_SIZE;
      const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

      for (int idx = tid; idx < B_sf_tile_size; idx += NUM_THREADS) {
        int n_local = idx / SF_PER_TILE;
        int sf_local = idx % SF_PER_TILE;
        int n_global = block_n + n_local;
        int k_block_global = k_block_start + sf_local;

        if (n_global < N && k_block_global < num_k_blocks) {
          B_sf_s[idx] =
              swizzle::read_scale(B_sf, n_global, k_block_global, N, K);
        } else {
          B_sf_s[idx] = 0;
        }
      }
    }

    __syncthreads();

// ----------------------------------------
// Compute: GDFS groups (zero-padded SMEM -> no per-element bounds check)
// ----------------------------------------
#pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        int local_m = thread_m + tm;
        int local_n = thread_n + tn;
        int out_row = block_m + local_m;
        int out_col = block_n + local_n;

        // Skip if out of bounds
        if (out_row >= M || out_col >= N) {
          continue;
        }

        // GDFS STP2-4: streaming group accumulate (no Product[GS] array),
        // then UE4M3 scale -> Operand. Zero-padded SMEM makes OOB elems
        // zero products, matching the original actual_groups/actual_k path.
        Operand tile_groups[GROUPS_PER_TILE];

#pragma unroll
        for (int grp = 0; grp < GROUPS_PER_TILE; grp++) {
          const DecodedFP4Frag a_frag{&As_dec[local_m * BKP + grp * GS]};
          const DecodedFP4Frag b_frag{&Bs_dec[local_n * BKP + grp * GS]};
          GroupResult gr =
              fp4_gdfs_group_accumulate_predecoded<GS>(a_frag, b_frag, g);
          if (gr.all_zero) {
            tile_groups[grp] = make_zero_operand();
          } else {
            int scale_idx = grp / GROUPS_PER_SCALE;
            uint8_t sfa = A_sf_s[local_m * SF_PER_TILE + scale_idx];
            uint8_t sfb = B_sf_s[local_n * SF_PER_TILE + scale_idx];
            tile_groups[grp] =
                apply_ue4m3_scales(gr.mantissa_sum, gr.max_exp, sfa, sfb, f, g);
          }
        }

        // STP5-7: Fused-sum accumulation over groups
        accum[tm][tn] = gdfs_accumulate_tile<GROUPS_PER_TILE>(tile_groups,
                                                              accum[tm][tn], f);
      }
    }

    __syncthreads();
  }

  // ========================================
  // EPILOGUE: Apply alpha and write output
  // ========================================
  const float alpha_val = alpha[0];

#pragma unroll
  for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_TILE_N; tn++) {
      int out_row = block_m + thread_m + tm;
      int out_col = block_n + thread_n + tn;

      if (out_row < M && out_col < N) {
        float scaled = accum[tm][tn] * alpha_val;
        C[out_row * N + out_col] = float_to_output_rn<OutDtype>(scaled);
      }
    }
  }
}

// ============================================================================
// MMA-Emu NVFP4 CoFDA GEMM Kernel
// ============================================================================

/**
 * @brief MMA-Emu NVFP4 CoFDA (Chunked FDA) GEMM kernel.
 *
 * Unlike GDFS, which groups products before applying scales, CoFDA applies
 * the UE4M3 block scale at the individual product level and then accumulates
 * in chunks of CS products.
 *
 * @tparam F Fractional bits F
 * @tparam CS CoFDA chunk size CS
 * @tparam OutDtype Output data type (__nv_bfloat16 or __half)
 */
template <int F, int CS, typename OutDtype>
__launch_bounds__(NVFP4EmuConfig::NUM_THREADS) __global__
    void mma_emu_scaled_nvfp4_mm_cofda_kernel(
        const uint8_t* __restrict__ A,     // [M, K/2] packed FP4
        const uint8_t* __restrict__ B,     // [N, K/2] packed FP4
        OutDtype* __restrict__ C,          // [M, N] output
        const uint8_t* __restrict__ A_sf,  // Swizzled block scales for A
                                           // (UE4M3)
        const uint8_t* __restrict__ B_sf,  // Swizzled block scales for B
                                           // (UE4M3)
        const float* __restrict__ alpha,   // [1] global scale
        const int M, const int N, const int K, const int f_bits) {
  const int f = (F == kRuntimeF) ? f_bits : F;

  // Import tiling constants
  constexpr int BM = NVFP4EmuConfig::BM;
  constexpr int BN = NVFP4EmuConfig::BN;
  constexpr int BK = NVFP4EmuConfig::BK;
  constexpr int BK_PACKED = NVFP4EmuConfig::BK_PACKED;
  constexpr int NUM_THREADS = NVFP4EmuConfig::NUM_THREADS;
  constexpr int THREAD_TILE_M = NVFP4EmuConfig::THREAD_TILE_M;
  constexpr int THREAD_TILE_N = NVFP4EmuConfig::THREAD_TILE_N;
  constexpr int BLOCK_SIZE = NVFP4EmuConfig::BLOCK_SIZE;
  constexpr int SF_PER_TILE = NVFP4EmuConfig::SF_PER_TILE;

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

  // ========================================
  // Shared Memory Declaration
  // ========================================
  __shared__ uint8_t As[NVFP4EmuConfig::SMEM_A_SIZE];
  __shared__ uint8_t Bs[NVFP4EmuConfig::SMEM_B_SIZE];
  __shared__ uint8_t A_sf_s[NVFP4EmuConfig::SMEM_A_SF_SIZE];
  __shared__ uint8_t B_sf_s[NVFP4EmuConfig::SMEM_B_SF_SIZE];

  // Thread tile mapping within block
  int thread_m, thread_n;
  get_thread_tile_position<NVFP4EmuConfig>(tid, thread_m, thread_n);

  // ========================================
  // Register Accumulators (FP32)
  // ========================================
  float accum[THREAD_TILE_M][THREAD_TILE_N];
#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; i++) {
#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j++) {
      accum[i][j] = 0.0f;
    }
  }

  // Precompute constants
  const int K_packed = K / 2;
  const int num_k_tiles = (K + BK - 1) / BK;

  // ========================================
  // OUTER LOOP: K-dimension tiles
  // ========================================
  for (int kt = 0; kt < num_k_tiles; kt++) {
    const int k_tile_start = kt * BK;
    const int k_tile_packed_start = k_tile_start / 2;
    const int k_block_start = kt * SF_PER_TILE;

    // ----------------------------------------
    // Load A tile: [BM, BK/2] from global to shared
    // ----------------------------------------
    {
      constexpr int A_tile_size = NVFP4EmuConfig::SMEM_A_SIZE;
      constexpr int A_elems_per_thread = NVFP4EmuConfig::A_ELEMS_PER_THREAD;

#pragma unroll
      for (int i = 0; i < A_elems_per_thread; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < A_tile_size) {
          int a_row = idx / BK_PACKED;
          int a_col = idx % BK_PACKED;
          int global_row = block_m + a_row;
          int global_col = k_tile_packed_start + a_col;

          if (global_row < M && global_col < K_packed) {
            As[idx] = A[global_row * K_packed + global_col];
          } else {
            As[idx] = static_cast<uint8_t>(__nv_fp4_storage_t(0));
          }
        }
      }
    }

    // ----------------------------------------
    // Load B tile: [BN, BK/2] from global to shared
    // ----------------------------------------
    {
      constexpr int B_tile_size = NVFP4EmuConfig::SMEM_B_SIZE;
      constexpr int B_elems_per_thread = NVFP4EmuConfig::B_ELEMS_PER_THREAD;

#pragma unroll
      for (int i = 0; i < B_elems_per_thread; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < B_tile_size) {
          int b_row = idx / BK_PACKED;
          int b_col = idx % BK_PACKED;
          int global_row = block_n + b_row;
          int global_col = k_tile_packed_start + b_col;

          if (global_row < N && global_col < K_packed) {
            Bs[idx] = B[global_row * K_packed + global_col];
          } else {
            Bs[idx] = static_cast<uint8_t>(__nv_fp4_storage_t(0));
          }
        }
      }
    }

    // ----------------------------------------
    // Load A scale factors (de-swizzle during load)
    // ----------------------------------------
    {
      constexpr int A_sf_tile_size = NVFP4EmuConfig::SMEM_A_SF_SIZE;
      const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

      for (int idx = tid; idx < A_sf_tile_size; idx += NUM_THREADS) {
        int m_local = idx / SF_PER_TILE;
        int sf_local = idx % SF_PER_TILE;
        int m_global = block_m + m_local;
        int k_block_global = k_block_start + sf_local;

        if (m_global < M && k_block_global < num_k_blocks) {
          A_sf_s[idx] =
              swizzle::read_scale(A_sf, m_global, k_block_global, M, K);
        } else {
          A_sf_s[idx] = 0;
        }
      }
    }

    // ----------------------------------------
    // Load B scale factors (de-swizzle during load)
    // ----------------------------------------
    {
      constexpr int B_sf_tile_size = NVFP4EmuConfig::SMEM_B_SF_SIZE;
      const int num_k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

      for (int idx = tid; idx < B_sf_tile_size; idx += NUM_THREADS) {
        int n_local = idx / SF_PER_TILE;
        int sf_local = idx % SF_PER_TILE;
        int n_global = block_n + n_local;
        int k_block_global = k_block_start + sf_local;

        if (n_global < N && k_block_global < num_k_blocks) {
          B_sf_s[idx] =
              swizzle::read_scale(B_sf, n_global, k_block_global, N, K);
        } else {
          B_sf_s[idx] = 0;
        }
      }
    }

    __syncthreads();

    // ----------------------------------------
    // Compute: CoFDA chunked accumulation
    // ----------------------------------------
    const int actual_k = min(BK, K - k_tile_start);

#pragma unroll
    for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
      for (int tn = 0; tn < THREAD_TILE_N; tn++) {
        int local_m = thread_m + tm;
        int local_n = thread_n + tn;
        int out_row = block_m + local_m;
        int out_col = block_n + local_n;

        // Skip if out of bounds
        if (out_row >= M || out_col >= N) {
          continue;
        }

        // ========================================
        // CoFDA: Chunk-based accumulation with product-level UE4M3 scales
        // ========================================
        for (int chunk_start = 0; chunk_start < actual_k; chunk_start += CS) {
          Operand chunk_operands[CS];

#pragma unroll
          for (int i = 0; i < CS; i++) {
            int k_local = chunk_start + i;
            if (k_local < actual_k) {
              uint8_t a_val =
                  load_fp4_from_tile(As, local_m, k_local, BK_PACKED);
              uint8_t b_val =
                  load_fp4_from_tile(Bs, local_n, k_local, BK_PACKED);

              // Scale mapping: k_local / BLOCK_SIZE gives scale index
              int scale_idx = k_local / BLOCK_SIZE;
              uint8_t sfa = A_sf_s[local_m * SF_PER_TILE + scale_idx];
              uint8_t sfb = B_sf_s[local_n * SF_PER_TILE + scale_idx];

              chunk_operands[i] = fp4_product_with_ue4m3_scales(
                  decompose_fp4_e2m1(a_val), decompose_fp4_e2m1(b_val), sfa,
                  sfb, f);
            } else {
              chunk_operands[i] = make_zero_operand();
            }
          }

          accum[tm][tn] =
              chunked_accumulate<CS>(chunk_operands, accum[tm][tn], f);
        }
      }
    }

    __syncthreads();
  }

  // ========================================
  // EPILOGUE: Apply alpha and write output
  // ========================================
  const float alpha_val = alpha[0];

#pragma unroll
  for (int tm = 0; tm < THREAD_TILE_M; tm++) {
#pragma unroll
    for (int tn = 0; tn < THREAD_TILE_N; tn++) {
      int out_row = block_m + thread_m + tm;
      int out_col = block_n + thread_n + tn;

      if (out_row < M && out_col < N) {
        float scaled = accum[tm][tn] * alpha_val;
        C[out_row * N + out_col] = float_to_output_rn<OutDtype>(scaled);
      }
    }
  }
}

// ============================================================================
// Kernel Dispatch
// ============================================================================
//
// NVFP4 fixes both CS and GS at 16, so the output dtype is the only template
// axis. F and G are kernel arguments.

namespace detail {

// F and G come from the lists in mma_emu_instantiation.cuh; an unlisted value
// falls through to the kRuntime instantiation rather than being refused.

#define VLLM_MMA_EMU_LAUNCH_NVFP4_GDFS(F_VAL, G_VAL)                        \
  mma_emu_scaled_nvfp4_mm_emu_kernel<F_VAL, G_VAL, 16, OutDtype>            \
      <<<grid, block, 0, stream>>>(a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, \
                                   alpha, M, N, K, f_bits, g_bits)

#define VLLM_MMA_EMU_NVFP4_G_CASE(G_VAL, F_VAL)   \
  if (g_bits == G_VAL) {                          \
    VLLM_MMA_EMU_LAUNCH_NVFP4_GDFS(F_VAL, G_VAL); \
    return;                                       \
  }

#define VLLM_MMA_EMU_NVFP4_GDFS_F_CASE(F_VAL, ...)            \
  if (f_bits == F_VAL) {                                      \
    VLLM_MMA_EMU_FP4_G_LIST(VLLM_MMA_EMU_NVFP4_G_CASE, F_VAL) \
    VLLM_MMA_EMU_LAUNCH_NVFP4_GDFS(F_VAL, kRuntimeG);         \
    return;                                                   \
  }

#define VLLM_MMA_EMU_LAUNCH_NVFP4_COFDA(F_VAL)                              \
  mma_emu_scaled_nvfp4_mm_cofda_kernel<F_VAL, 16, OutDtype>                 \
      <<<grid, block, 0, stream>>>(a_ptr, b_ptr, c_ptr, a_sf_ptr, b_sf_ptr, \
                                   alpha, M, N, K, f_bits)

#define VLLM_MMA_EMU_NVFP4_COFDA_F_CASE(F_VAL, ...) \
  if (f_bits == F_VAL) {                            \
    VLLM_MMA_EMU_LAUNCH_NVFP4_COFDA(F_VAL);         \
    return;                                         \
  }

template <typename OutDtype>
inline void dispatch_nvfp4(dim3 grid, dim3 block, cudaStream_t stream,
                           const void* a, const void* b, void* out,
                           const void* a_sf, const void* b_sf,
                           const float* alpha, int M, int N, int K,
                           int algorithm, int f_bits, int g_bits) {
  const auto* a_ptr = static_cast<const uint8_t*>(a);
  const auto* b_ptr = static_cast<const uint8_t*>(b);
  const auto* a_sf_ptr = static_cast<const uint8_t*>(a_sf);
  const auto* b_sf_ptr = static_cast<const uint8_t*>(b_sf);
  auto* c_ptr = static_cast<OutDtype*>(out);

  if (algorithm == design_space::kGDFS) {
    VLLM_MMA_EMU_F_LIST(VLLM_MMA_EMU_NVFP4_GDFS_F_CASE)
    VLLM_MMA_EMU_LAUNCH_NVFP4_GDFS(kRuntimeF, kRuntimeG);
  } else {
    VLLM_MMA_EMU_F_LIST(VLLM_MMA_EMU_NVFP4_COFDA_F_CASE)
    VLLM_MMA_EMU_LAUNCH_NVFP4_COFDA(kRuntimeF);
  }
}

#undef VLLM_MMA_EMU_NVFP4_COFDA_F_CASE
#undef VLLM_MMA_EMU_LAUNCH_NVFP4_COFDA
#undef VLLM_MMA_EMU_NVFP4_GDFS_F_CASE
#undef VLLM_MMA_EMU_NVFP4_G_CASE
#undef VLLM_MMA_EMU_LAUNCH_NVFP4_GDFS

}  // namespace detail

// ============================================================================
// Host Entry
// ============================================================================
//
// Takes raw pointers rather than tensors: the torch boundary lives entirely in
// mma_emu_scaled_nvfp4_mm_entry.cu, which validates every argument before
// calling here.

inline void mma_emu_scaled_nvfp4_mm_emu(void* out, const void* a, const void* b,
                                        const void* a_sf, const void* b_sf,
                                        const float* alpha, int M, int N, int K,
                                        design_space::MmaEmuOutDtype out_dtype,
                                        int algorithm, int f_bits, int g_bits,
                                        cudaStream_t stream) {
  using Config = NVFP4EmuConfig;
  dim3 grid((N + Config::BN - 1) / Config::BN,
            (M + Config::BM - 1) / Config::BM);
  dim3 block(Config::NUM_THREADS);

  if (out_dtype == design_space::MmaEmuOutDtype::kBFloat16) {
    detail::dispatch_nvfp4<__nv_bfloat16>(grid, block, stream, a, b, out, a_sf,
                                          b_sf, alpha, M, N, K, algorithm,
                                          f_bits, g_bits);
  } else {
    detail::dispatch_nvfp4<__half>(grid, block, stream, a, b, out, a_sf, b_sf,
                                   alpha, M, N, K, algorithm, f_bits, g_bits);
  }
}

}  // namespace mma_emu
}  // namespace vllm
