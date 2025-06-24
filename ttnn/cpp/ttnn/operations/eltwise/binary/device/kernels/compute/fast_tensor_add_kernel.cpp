// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compile_time_args.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "debug/dprint.h"

using namespace std;

namespace NAMESPACE
{
void MAIN {

    uint32_t input0_cb_index = get_arg_val<uint32_t>(0);
    uint32_t input1_cb_index = get_arg_val<uint32_t>(1);
    uint32_t output_cb_index = get_arg_val<uint32_t>(2);
    
    init_sfpu(input0_cb_index, output_cb_index);

    cb_reserve_back(output_cb_index, 1);

    cb_wait_front(input0_cb_index, 1);
    cb_wait_front(input1_cb_index, 1);
    DPRINT << "(compute) got tiles in input cb(s)" << ENDL();

    tile_regs_acquire();
    DPRINT << "(compute) dst reg acquired " << ENDL();

    copy_tile_to_dst_init_short(input0_cb_index);
    copy_tile(input0_cb_index, 0, 0);

    copy_tile_to_dst_init_short(input1_cb_index);
    copy_tile(input1_cb_index, 0, 1);

    add_binary_tile_init();
    add_binary_tile(0, 1);

    tile_regs_commit();

    tile_regs_wait();

    pack_tile(0, output_cb_index);

    tile_regs_release();

    cb_push_back(output_cb_index, 1);
    cb_pop_front(input0_cb_index, 1);
    cb_pop_front(input1_cb_index, 1);

    DPRINT << "(compute) finished" << ENDL();
}
}