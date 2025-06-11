
#include "dataflow_api.h"
#include "compile_time_args.h"

#include <cstdint>

using namespace std;

void write_cb_page_into_dram_page(uint32_t bank_id, uint32_t bank_addr_offset, uint32_t cb_index)
{
	// Wait for 1 tile to be available
	cb_wait_front(cb_index, 1);
	uint32_t cb_l1_address = get_read_ptr(cb_index);

	uint64_t dram_noc_address = get_noc_addr_from_bank_id<true>(bank_id, bank_addr_offset);
	uint32_t cb_page_size = get_tile_size(cb_index);
	noc_async_write(cb_l1_address, dram_noc_address, cb_page_size);

	// Remove the tile to make space for another tile
	cb_pop_front(cb_index, 1);
	DPRINT << "(writer) cb_pop_front" << ENDL();
}

void kernel_main()
{
	// Writer kernel for add_2_integers_in_SFPU_rpsingh.cpp

	uint32_t input_num_elements = get_compile_time_arg_val(0);
	uint32_t dram_page_count = get_compile_time_arg_val(1);
	
	uint32_t output_cb_index = get_compile_time_arg_val(2);
	uint32_t output_dram_address = get_compile_time_arg_val(3);

	const uint32_t num_ints_per_dram_page = input_num_elements / dram_page_count;
	const uint32_t dram_page_size = num_ints_per_dram_page * sizeof(uint32_t);
	
	uint32_t cb_page_size = get_tile_size(output_cb_index);

	for(uint32_t i = 0; i < dram_page_count; ++i)
	{
		uint32_t write_bytes = 0;
		while(write_bytes < dram_page_size)
		{
			write_cb_page_into_dram_page(i, output_dram_address + write_bytes, output_cb_index);
			write_bytes += cb_page_size;
		}
	}

	noc_async_write_barrier();
	DPRINT << "writer finished" << ENDL();
}