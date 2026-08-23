/*
 * MMA-Emu compile-time instantiation lists
 *
 * F and G reach the kernels as literals when they appear here, and as ordinary
 * arguments when they do not. The dispatch tries this list first and falls back
 * to the kRuntime instantiation, so the list is a specialization cache rather
 * than a statement of what is supported -- design_space.cuh remains the
 * authority on that, and every value it accepts still runs.
 *
 * The cost of an entry is one kernel instantiation per (CS or GS) x dtype, and
 * the lists multiply: an F here costs two CoFDA instantiations, while an F and
 * a G together cost four GDFS ones. Keep that in view before widening either.
 */

// clang-format off
#pragma once

#include "design_space.cuh"

namespace vllm {
namespace mma_emu {

// Sentinels meaning "read it from the kernel argument". Outside the accepted
// ranges on purpose, so they cannot collide with a real F or G.
inline constexpr int kRuntimeF = -1;
inline constexpr int kRuntimeG = -1;

}  // namespace mma_emu
}  // namespace vllm

// F, over design_space::F_MIN .. F_MAX. Shared by CoFDA and the GDFS
// inter-group stage, and by both formats.
#define VLLM_MMA_EMU_F_LIST(APPLY, ...) \
  APPLY( 7, __VA_ARGS__) APPLY( 8, __VA_ARGS__) APPLY( 9, __VA_ARGS__) \
  APPLY(10, __VA_ARGS__) APPLY(11, __VA_ARGS__) APPLY(12, __VA_ARGS__) \
  APPLY(13, __VA_ARGS__) APPLY(14, __VA_ARGS__) APPLY(15, __VA_ARGS__) \
  APPLY(16, __VA_ARGS__) APPLY(17, __VA_ARGS__) APPLY(18, __VA_ARGS__) \
  APPLY(19, __VA_ARGS__) APPLY(20, __VA_ARGS__) APPLY(21, __VA_ARGS__) \
  APPLY(22, __VA_ARGS__) APPLY(23, __VA_ARGS__) APPLY(24, __VA_ARGS__) \
  APPLY(25, __VA_ARGS__) APPLY(26, __VA_ARGS__) APPLY(27, __VA_ARGS__) \
  APPLY(28, __VA_ARGS__) APPLY(29, __VA_ARGS__) APPLY(30, __VA_ARGS__) \
  APPLY(31, __VA_ARGS__) APPLY(32, __VA_ARGS__) APPLY(33, __VA_ARGS__) \
  APPLY(34, __VA_ARGS__) APPLY(35, __VA_ARGS__)

// G for FP8 GDFS, matching design_space::FP8_G.
#define VLLM_MMA_EMU_FP8_G_LIST(APPLY, ...) \
  APPLY( 3, __VA_ARGS__) APPLY( 4, __VA_ARGS__) APPLY( 5, __VA_ARGS__) \
  APPLY( 6, __VA_ARGS__) APPLY(32, __VA_ARGS__)

// G for NVFP4 GDFS, over design_space::G_MIN .. FP4_G_MAX.
#define VLLM_MMA_EMU_FP4_G_LIST(APPLY, ...) \
  APPLY( 3, __VA_ARGS__) APPLY( 4, __VA_ARGS__) APPLY( 5, __VA_ARGS__) \
  APPLY( 6, __VA_ARGS__)
// clang-format on
