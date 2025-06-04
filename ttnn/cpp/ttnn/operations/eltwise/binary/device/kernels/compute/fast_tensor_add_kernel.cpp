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
    uint32_t src0_addr = get_compile_time_arg_val(0);
    uint32_t src1_addr = get_compile_time_arg_val(1);
    uint32_t dst_addr = get_compile_time_arg_val(2);

    volatile uint32_t* src0_ptr = (uint32_t*)(src0_addr);
    volatile uint32_t* src1_ptr = (uint32_t*)(src1_addr);

    volatile uint32_t* dst_ptr = (uint32_t*)(dst_addr);

    uint32_t num_elements = get_compile_time_arg_val(3);
    for(uint32_t i = 0; i < num_elements; i++)
    	dst_ptr[i] = src0_ptr[i] + src1_ptr[i];
}
}