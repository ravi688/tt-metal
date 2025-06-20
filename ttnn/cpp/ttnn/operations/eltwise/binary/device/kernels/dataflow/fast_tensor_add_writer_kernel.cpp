
#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cstdint>
#include <cstring>

void kernel_main()
{
	uint32_t output_cb_index = get_arg_val<uint32_t>(0);
	uint32_t output_l1_addr = get_arg_val<uint32_t>(1);
	uint32_t num_columns = get_arg_val<uint32_t>(2);
	uint32_t page_size = get_arg_val<uint32_t>(3);
	uint32_t bank_id = get_arg_val<uint32_t>(4);

	cb_wait_front(output_cb_index, 1);

	DPRINT << "(writer) got a tile in output cb" << ENDL();

	uint32_t output_ptr = get_read_ptr(output_cb_index);
	uint32_t size = num_columns * 4;

	uint64_t output_l1_noc_addr = get_l1_noc_addr(bank_id, page_size, output_l1_addr);
	noc_async_write(output_ptr, output_l1_noc_addr, page_size);

	DPRINT << "(writer) did memcpy " << ENDL();

	cb_pop_front(output_cb_index, 1);

	DPRINT << "(writer) finished" << ENDL();
}
