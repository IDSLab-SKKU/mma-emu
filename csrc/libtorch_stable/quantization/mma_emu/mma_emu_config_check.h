/*
 * MMA-Emu configuration validation
 *
 * Host-only. One function decides whether an (algorithm, F, G, GS, CS) tuple
 * is accepted, and phrases the rejection. Both the operator entry points and
 * the Python kernel selector go through it, so the accepted values and the
 * message describing them are stated once, here and in design_space.cuh.
 */

#pragma once

#include <iterator>
#include <string>

#include "design_space.cuh"

namespace vllm {
namespace mma_emu {

/**
 * @brief Check an emulation configuration.
 *
 * @param algorithm design_space::Algorithm
 * @param f_bits fractional bits F
 * @param g_bits GDFS intra-group bits G (ignored unless algorithm is GDFS)
 * @param group_size GDFS group size GS (ignored for NVFP4 and for CoFDA)
 * @param chunk_size CoFDA chunk size CS (ignored for NVFP4 and for GDFS)
 * @param is_fp4 true for NVFP4, false for FP8
 * @return empty string if the configuration is accepted, otherwise the reason
 */
inline std::string mma_emu_config_error(int64_t algorithm, int64_t f_bits,
                                        int64_t g_bits, int64_t group_size,
                                        int64_t chunk_size, bool is_fp4) {
  namespace ds = design_space;
  const auto num = [](int64_t v) { return std::to_string(v); };
  const auto range = [&num](int64_t lo, int64_t hi) {
    return "[" + num(lo) + ", " + num(hi) + "]";
  };
  // Spelled from the set itself, so the accepted values are stated once.
  const auto set = [&num](const auto& values) {
    std::string out = "{";
    for (size_t i = 0; i < std::size(values); ++i) {
      out += (i ? ", " : "") + num(values[i]);
    }
    return out + "}";
  };

  if (algorithm != ds::kGDFS && algorithm != ds::kCoFDA) {
    return "algorithm must be " + num(ds::kGDFS) + " (GDFS) or " +
           num(ds::kCoFDA) + " (CoFDA), got " + num(algorithm);
  }

  if (!ds::in_range(ds::F_MIN, ds::F_MAX, f_bits)) {
    return "f_bits must be in " + range(ds::F_MIN, ds::F_MAX) + ", got " +
           num(f_bits);
  }

  if (algorithm == ds::kGDFS) {
    // FP4's accepted G is a contiguous range; FP8's is an enumerated set.
    const bool g_ok = is_fp4 ? ds::in_range(ds::G_MIN, ds::FP4_G_MAX, g_bits)
                             : ds::contains(ds::FP8_G, g_bits);
    if (!g_ok) {
      return std::string("g_bits must be in ") +
             (is_fp4 ? range(ds::G_MIN, ds::FP4_G_MAX) : set(ds::FP8_G)) +
             " for " + (is_fp4 ? "NVFP4" : "FP8") + ", got " + num(g_bits);
    }
    // NVFP4 fixes GS at 16 in the kernel, so it is not a caller choice.
    if (!is_fp4 && !ds::contains(ds::FP8_GS, group_size)) {
      return "group_size must be 8 or 16, got " + num(group_size);
    }
  } else if (!is_fp4 && !ds::contains(ds::FP8_CS, chunk_size)) {
    // NVFP4 fixes CS at 16 in the kernel.
    return "chunk_size must be 16 or 32, got " + num(chunk_size);
  }

  return {};
}

}  // namespace mma_emu
}  // namespace vllm
