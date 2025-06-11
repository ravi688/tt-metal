#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include <cstdint>

using namespace std;

namespace NAMESPACE
{
	void MAIN
	{
		// Compute Kernel for add_2_integers_in_SFPU_rpsingh.cpp

		uint32_t input_num_elements = get_compile_time_arg_val(0);
		uint32_t input0_cb_index = get_compile_time_arg_val(1);
		uint32_t input1_cb_index = get_compile_time_arg_val(2);
		uint32_t output_cb_index = get_compile_time_arg_val(3);

		// Runs parts in MATH, PACK, and UNPACK
		binary_op_init_common(input0_cb_index, input1_cb_index, output_cb_index);
		// Runs parts in MATH, and UNPACK
		add_tiles_init(input0_cb_index, input1_cb_index);

		uint32_t i = 0;
		while(i < input_num_elements)
		{
			// Wait for the writer kernel to read and pop out to a tile
			cb_reserve_back(output_cb_index, 1);

			// Wait for the reader kernel to write and push back a tile in input0 cb, and input1 cb
			cb_wait_front(input0_cb_index, 1);
			cb_wait_front(input1_cb_index, 1);

			// Runs in MATH
			tile_regs_acquire();

			// Runs parts in UNPACK, and MATH
			add_tiles(input0_cb_index, input1_cb_index, i, i, i);

			// Runs in MATH
			tile_regs_commit();
			// Runs in PACK
			pack_tile(i, output_cb_index);
			// Runs in PACK
			tile_regs_release();

			// Remove one tile from each input circular buffer and let the reader kernel to write another tile into these cb(s).
			cb_pop_front(input0_cb_index, 1);
			cb_pop_front(input1_cb_index, 1);

			// Add tile and let the writer kernel to read this tile
			cb_push_back(output_cb_index, 1);

			++i;
		}

	}
}