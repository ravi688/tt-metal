
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

	uint32_t page_size = get_arg_val<uint32_t>(6);
	uint32_t bank_id = get_arg_val<uint32_t>(7);

	DPRINT << "num_coulmns: " << num_columns << ENDL();

	float myFloat = 3.4f;
	myFloat *= 2.0f;
	DPRINT << "float support: "<< myFloat << ENDL();


	DPRINT << "(reader) trying to reserve tiles in input cb(s) " << ENDL();

	cb_reserve_back(input0_cb_index, 1);
	cb_reserve_back(input1_cb_index, 1);

	DPRINT << "(reader) reserved tiles in input cb(s) " << ENDL();

	uint32_t input0_ptr = get_write_ptr(input0_cb_index);
	uint32_t input1_ptr = get_write_ptr(input1_cb_index);

	uint64_t input0_l1_noc_addr = get_l1_noc_addr(bank_id, page_size, input0_l1_addr);
	uint64_t input1_l1_noc_addr = get_l1_noc_addr(bank_id, page_size, input1_l1_addr);

	uint32_t size = num_columns * 4;

	noc_async_read(input0_l1_noc_addr, input0_ptr, page_size);
	noc_async_read(input1_l1_noc_addr, input1_ptr, page_size);

	DPRINT << "input0 row: ";
	float* flts0 = reinterpret_cast<float*>(input0_ptr);
	for(uint32_t i = 0; i < num_columns; ++i)
		DPRINT << flts0[i] << " ";
	DPRINT << ENDL();

	DPRINT << "input1 row: ";
	float* flts1 = reinterpret_cast<float*>(input1_ptr);
	for(uint32_t i = 0; i < num_columns; ++i)
		DPRINT << flts1[i] << " ";
	DPRINT << ENDL();

	DPRINT << "(reader) did memcpy " << ENDL();

	cb_push_back(input0_cb_index, 1);
	cb_push_back(input1_cb_index, 1);

	DPRINT << "(reader) finished" << ENDL();
}
