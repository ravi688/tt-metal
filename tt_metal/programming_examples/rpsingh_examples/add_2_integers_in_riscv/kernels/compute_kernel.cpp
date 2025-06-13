#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/add_uint32_sfpu.h"

#include <cstdint>

using namespace std;

namespace NAMESPACE
{
	void MAIN
	{
		// Compute Kernel for add_2_integers_in_SFPU_rpsingh.cpp

		uint32_t input_num_elements = get_compile_time_arg_val(0);
		uint32_t cb_page_size = get_compile_time_arg_val(1);
		uint32_t input0_cb_index = get_compile_time_arg_val(2);
		uint32_t input1_cb_index = get_compile_time_arg_val(3);
		uint32_t output_cb_index = get_compile_time_arg_val(4);

		uint32_t num_cb_pages_to_read = (input_num_elements * sizeof(uint32_t)) /  cb_page_size;
		uint32_t i = 0;

		unary_op_init_common(input0_cb_index, input1_cb_index);

		while(i < num_cb_pages_to_read)
		{
			// Wait for the writer kernel to read and pop out to a tile
			cb_reserve_back(output_cb_index, 1);

			// Wait for the reader kernel to write and push back a tile in input0 cb, and input1 cb
			cb_wait_front(input0_cb_index, 1);
			cb_wait_front(input1_cb_index, 1);

			// acquire tile registers
    		tile_regs_acquire();

    		copy_tile_to_dst_init_short(input0_cb_index);
    		copy_tile(input0_cb_index, 0, 0);

    		copy_tile_to_dst_init_short(input1_cb_index);
    		copy_tile(input1_cb_index, 0, 1);

			add_uint32_tile_init();
    		add_uint32_tile(0, 1);

    		 // signal the packer
    		tile_regs_commit();

    		// packer waits here
    		tile_regs_wait(); 
    		pack_tile(0, output_cb_index);

    		// release tile registers
    		tile_regs_release();


			// Remove one tile from each input circular buffer and let the reader kernel to write another tile into these cb(s).
			cb_pop_front(input0_cb_index, 1);
			cb_pop_front(input1_cb_index, 1);

			// Add tile and let the writer kernel to read this tile
			cb_push_back(output_cb_index, 1);

			++i;
		}
		DPRINT << "compute finished" << ENDL();
	}
}