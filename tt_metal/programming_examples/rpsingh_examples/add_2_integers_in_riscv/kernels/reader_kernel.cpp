
#include "dataflow_api.h"
#include "compile_time_args.h"

#include <cstdint>

using namespace std;

void read_dram_page_into_cb_page(uint32_t bank_id, uint32_t bank_addr_offset, uint32_t cb_index)
{
	// Reserve 1 tile on the back of the circular buffer
	cb_reserve_back(cb_index, 1);
	// and get pointer to the tile
	uint32_t cb_l1_address = get_write_ptr(cb_index);

	uint64_t dram_noc_address = get_noc_addr_from_bank_id<true>(bank_id, bank_addr_offset);
	uint32_t cb_page_size = get_tile_size(cb_index);
	noc_async_read(dram_noc_address, cb_l1_address, cb_page_size);
	noc_async_read_barrier();
	
	// Make this tile visible to a reader (via cb_wait_front())
	cb_push_back(cb_index, 1);
}

void kernel_main()
{
	// Reader Kernel for add_2_integers_in_SFPU_rpsingh.cpp

	uint32_t input_num_elements = get_compile_time_arg_val(0);
	uint32_t dram_page_count = get_compile_time_arg_val(1);

	uint32_t input0_cb_index = get_compile_time_arg_val(2);
	uint32_t input1_cb_index = get_compile_time_arg_val(3);
	uint32_t input0_dram_address = get_compile_time_arg_val(4);
	uint32_t input1_dram_address = get_compile_time_arg_val(5);

	const uint32_t num_ints_per_dram_page = input_num_elements / dram_page_count;
	const uint32_t dram_page_size = num_ints_per_dram_page * sizeof(uint32_t);

	uint32_t cb_page_size = get_tile_size(input0_cb_index);
	
	for(uint32_t i = 0; i < dram_page_count; ++i)
	{
		uint32_t read_bytes = 0;
		while(read_bytes < dram_page_size)
		{
			read_dram_page_into_cb_page(i, input0_dram_address + read_bytes, input0_cb_index);
			read_dram_page_into_cb_page(i, input1_dram_address + read_bytes, input1_cb_index);
			read_bytes += cb_page_size;
		}
	}
	DPRINT << "reader finished" << ENDL();
}