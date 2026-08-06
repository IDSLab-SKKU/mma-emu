/*
 * MMA Arithmetic Design Space
 *
 * Single source of truth for the parameter values the kernels accept.
 *
 * F and G are kernel arguments, so they are bounded by a range. CS and GS
 * select a template instantiation, so they are enumerated: the dispatch
 * instantiates one template per (CS, GS, output dtype) combination, and
 * bounding those sets bounds the build.
 *
 * The accumulation configurations of the commercially deployed architectures
 * all fall inside these sets: Ada (CoFDA, CS=16, F=13), Hopper (FDA, CS=32,
 * F=13), Blackwell FP8 (FDA, CS=32, F=25), and Blackwell NVFP4 (GDFS, G=6,
 * F=35, GS=16).
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace vllm {
namespace mma_emu {
namespace design_space {

// Accumulation algorithm, as passed to the torch operators.
enum Algorithm : int64_t {
  kGDFS = 1,   // Group-Dot-Fused-Sum
  kCoFDA = 2,  // Chain-of-FDA, C-fused accumulator
};

// Fractional bits F, shared by CoFDA and the GDFS inter-group stage.
inline constexpr int64_t F_MIN = 7;
inline constexpr int64_t F_MAX = 35;

// Intra-group fractional bits G for GDFS. The upper bounds are the lossless
// widths of the respective products: past them the intra-group sum carries no
// further information, so raising G has no effect.
inline constexpr int64_t G_MIN = 3;
inline constexpr int64_t FP8_G_MAX = 32;  // FP8 E4M3 products
inline constexpr int64_t FP4_G_MAX = 6;   // E2M1 products

// Chunk size CS for CoFDA. NVFP4 is fixed at CS = 16 in the kernel.
inline constexpr int64_t FP8_CS[] = {16, 32};

// Group size GS for GDFS. NVFP4 is fixed at GS = 16 in the kernel.
inline constexpr int64_t FP8_GS[] = {8, 16};

constexpr bool in_range(int64_t lo, int64_t hi, int64_t value) {
  return lo <= value && value <= hi;
}

template <size_t N>
constexpr bool contains(const int64_t (&set)[N], int64_t value) {
  for (size_t i = 0; i < N; ++i) {
    if (set[i] == value) return true;
  }
  return false;
}

}  // namespace design_space
}  // namespace mma_emu
}  // namespace vllm
