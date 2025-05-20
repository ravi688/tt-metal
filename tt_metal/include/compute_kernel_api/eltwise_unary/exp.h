// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_exp.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
template <bool approx = true, bool fast_and_approx = true, uint32_t scale = 0x3F800000>
ALWI void exp_tile_init() {
    MATH((llk_math_eltwise_unary_sfpu_exponential_init<approx, fast_and_approx, scale>()));
}

// clang-format off
/**
 * Performs element-wise computation of exponential on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
// clang-format on
template <
    bool approx = true,
    bool fast_and_approx = true,
    int iterations = 8,
    bool scale_en = false,
    bool skip_positive_check = false>
ALWI void exp_tile(uint32_t idst, int vector_mode = (int)VectorMode::RC, uint16_t scale = 0x3F80) {
    MATH((llk_math_eltwise_unary_sfpu_exponential<approx, fast_and_approx, iterations, scale_en, skip_positive_check>(
        idst, vector_mode, iterations, scale)));
}

}  // namespace ckernel
