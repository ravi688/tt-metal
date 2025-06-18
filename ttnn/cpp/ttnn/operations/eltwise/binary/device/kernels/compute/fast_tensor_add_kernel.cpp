// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compile_time_args.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "debug/dprint.h"

using namespace std;

namespace NAMESPACE
{
void MAIN {

    uint32_t input0_cb_index = get_arg_val<uint32_t>(0);
    uint32_t input1_cb_index = get_arg_val<uint32_t>(1);
    uint32_t output_cb_index = get_arg_val<uint32_t>(2);
    
    binary_op_init_common(input0_cb_index, input1_cb_index, output_cb_index);

    add_tiles_init(input0_cb_index, input1_cb_index);

    acquire_dst();

    cb_wait_front(input0_cb_index, 1);
    cb_wait_front(input1_cb_index, 1);

    add_tiles(input0_cb_index, input1_cb_index, 0, 0, 0);

    pack_tile(0, output_cb_index);

    cb_push_back(output_cb_index, 1);
    cb_pop_front(input0_cb_index, 1);
    cb_pop_front(input1_cb_index, 1);

    release_dst();

    DPRINT << "(compute) finished" << END();
}
}