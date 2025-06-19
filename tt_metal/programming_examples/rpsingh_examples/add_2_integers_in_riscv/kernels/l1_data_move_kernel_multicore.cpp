
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main()
{
	// Number of pages in which the inputs and outputs are divided
	uint32_t page_count = get_compile_time_arg_val(0);
	// element counts in the input vectors
	uint32_t input_size = get_compile_time_arg_val(1);
	uint32_t input0_l1_address = get_compile_time_arg_val(2);
	uint32_t input1_l1_address = get_compile_time_arg_val(3);
	uint32_t output_l1_address = get_compile_time_arg_val(4);

	DPRINT << "input0_l1_address: "  << input0_l1_address << ENDL();
	DPRINT << "input1_l1_address: "  << input1_l1_address << ENDL();
	DPRINT << "output_l1_address: "  << output_l1_address << ENDL();


	uint32_t per_core_input_size = input_size / page_count;

	// L1 Pointers are directly accessible to Baby RISC-V cores
	uint32_t* input0_l1_ptr = (uint32_t*)input0_l1_address;
	uint32_t* input1_l1_ptr = (uint32_t*)input1_l1_address;
	uint32_t* output_l1_ptr = (uint32_t*)output_l1_address;

	DPRINT << "input0_l1_ptr: " << input0_l1_ptr[0] << ENDL();
	DPRINT << "input1_l1_ptr: " << input1_l1_ptr[0] << ENDL();
	DPRINT << "output_l1_ptr: " << output_l1_ptr[0] << ENDL();

	// Perform Addition
	for(uint32_t i = 0; i < per_core_input_size; ++i)
		output_l1_ptr[i] = input0_l1_ptr[i] + input1_l1_ptr[i];
}
