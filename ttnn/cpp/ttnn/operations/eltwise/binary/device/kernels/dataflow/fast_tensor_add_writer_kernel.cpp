
#include "dataflow_api.h"

#include <cstdint>

void kernel_main()
{
	uint32_t output_cb_index = get_arg_val<uint32_t>(0);
	uint32_t output_l1_addr = get_arg_val<uint32_t>(1);
	uint32_t num_columns = get_arg_val<uint32_t>(2);

	cb_wait_front(output_cb_index, 1);

	uint32_t output_ptr = get_read_ptr(output_cb_index);
	uint32_t size = num_columns * 4;
	std::memcpy((void*)output_l1_addr, (void*)output_ptr, size);

	cb_pop_front(output_cb_index, 1);
}
