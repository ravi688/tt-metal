
#include "dataflow_api.h"

#include <cstdint>

void kernel_main()
{
	// Circular buffer indices
	uint32_t input0_cb_index = get_arg_val<uint32_t>(0);
	uint32_t input1_cb_index = get_arg_val<uint32_t>(1);
	uint32_t output_cb_index = get_arg_val<uint32_t>(2);

	// L1 addresses
	uint32_t input0_l1_addr = get_arg_val<uint32_t>(3);
	uint32_t input1_l1_addr = get_arg_val<uint32_t>(4);

	uint32_t num_columns = get_arg_val<uint32_t>(5);

	cb_reserve_back(input0_cb_index, 1);
	cb_reserve_back(input1_cb_index, 2);

	uint32_t input0_ptr = get_write_ptr(input0_cb_index);
	uint32_t input1_ptr = get_write_ptr(input1_cb_index);

	uint32_t size = num_columns * 4;
	std::memcpy((void*)input0_ptr, (void*)input0_l1_addr, size);
	std::memcpy((void*)input1_ptr, (void*)input1_l1_addr, size);

	cb_push_back(input0_cb_index, 1);
	cb_push_back(input1_cb_index, 2);
}
