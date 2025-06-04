// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compile_time_args.h"
#include "compute_kernel_api.h"

using namespace std;

namespace NAMESPACE
{
void MAIN {
    constexpr uint32_t src0_addr = get_compile_time_arg_val(0);
    constexpr uint32_t src1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t dst_addr = get_compile_time_arg_val(2);

    volatile int32_t* src0_ptr = (int32_t*)(src0_addr);
    volatile int32_t* src1_ptr = (int32_t*)(src1_addr);

    int32_t* dst_ptr = (int32_t*)(dst_addr);

    constexpr int32_t num_elements = get_compile_time_arg_val(3);
    for(int32_t i = 0; i < num_elements; i++)
        dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
}
}