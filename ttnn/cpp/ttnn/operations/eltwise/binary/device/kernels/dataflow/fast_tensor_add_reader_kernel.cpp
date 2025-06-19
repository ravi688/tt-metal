
#include "dataflow_api.h"
#include "debug/dprint.h"

#include <cstdint>
#include <cstring>

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

	DPRINT << "num_coulmns: " << num_columns << ENDL();

	DPRINT << "input0 row: ";
	float* flts0 = reinterpret_cast<float*>(input0_l1_addr);
	for(uint32_t i = 0; i < num_columns; ++i)
		DPRINT << flts0[i] << " ";
	DPRINT << ENDL();

	DPRINT << "input1 row: ";
	float* flts1 = reinterpret_cast<float*>(input1_l1_addr);
	for(uint32_t i = 0; i < num_columns; ++i)
		DPRINT << flts1[i] << " ";
	DPRINT << ENDL();

	DPRINT << "(reader) trying to reserve tiles in input cb(s) " << ENDL();

	cb_reserve_back(input0_cb_index, 1);
	cb_reserve_back(input1_cb_index, 1);

	DPRINT << "(reader) reserved tiles in input cb(s) " << ENDL();

	uint32_t input0_ptr = get_write_ptr(input0_cb_index);
	uint32_t input1_ptr = get_write_ptr(input1_cb_index);

	uint32_t size = num_columns * 4;

	// TODO: Replace this with noc based copy
	std::memcpy((void*)input0_ptr, (void*)input0_l1_addr, size);
	// TODO: Replace this with noc based copy
	std::memcpy((void*)input1_ptr, (void*)input1_l1_addr, size);

	DPRINT << "(reader) did memcpy " << ENDL();

	cb_push_back(input0_cb_index, 1);
	cb_push_back(input1_cb_index, 1);

	DPRINT << "(reader) finished" << ENDL();
}
