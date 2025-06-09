
#include "dataflow_api.h"
#include "debug/dprint.h"

// On wormhole chip we have 12 DRAM banks
static constexpr uint32_t DRAM_BANK_COUNT = 12;

void read_dram_pages(uint32_t src_dram_address, uint32_t l1_addr, uint32_t page_count, uint32_t page_size)
{
	DPRINT << "src_dram_address: " << src_dram_address << ENDL();
	for(uint32_t i = 0; i < page_count; ++i)
	{
		uint32_t bank_id = page_count % DRAM_BANK_COUNT;
		auto dram_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, src_dram_address);
		DPRINT << dram_noc_addr << ENDL();
		noc_async_read(dram_noc_addr, l1_addr + i * page_size, page_size);
		noc_async_read_barrier();
	}
}

void write_dram_pages(uint32_t dst_dram_address, uint32_t l1_addr, uint32_t page_count, uint32_t page_size)
{
	for(uint32_t i = 0; i < page_count; ++i)
	{
		uint32_t bank_id = page_count % DRAM_BANK_COUNT;
		auto dram_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dst_dram_address);
		noc_async_write(l1_addr + i * page_size, dram_noc_addr, page_size);
		noc_async_write_barrier();
	}
}

void kernel_main()
{
	// Number of pages in which the inputs and outputs are divided
	uint32_t page_count = get_compile_time_arg_val(0);
	// L1 base address
	uint32_t l1_mem_addr = get_compile_time_arg_val(1);
	// element counts in the input vectors
	uint32_t input_size = get_compile_time_arg_val(2);
	uint32_t input0_dram_address = get_compile_time_arg_val(3);
	uint32_t input1_dram_address = get_compile_time_arg_val(4);
	uint32_t output_dram_address = get_compile_time_arg_val(5);


	uint32_t input_size_in_bytes = input_size * sizeof(uint32_t);
	uint32_t input0_l1_address = l1_mem_addr;
	uint32_t input1_l1_address = l1_mem_addr + input_size_in_bytes;
	uint32_t output_l1_address = input1_l1_address + input_size_in_bytes;

	uint32_t page_size = input_size_in_bytes / page_count;

	// Read input DRAM buffers into L1
	read_dram_pages(input0_dram_address, input0_l1_address, page_count, page_size);
	read_dram_pages(input1_dram_address, input1_l1_address, page_count, page_size);

	uint32_t* input0_l1_ptr = (uint32_t*)input0_l1_address;
	uint32_t* input1_l1_ptr = (uint32_t*)input1_l1_address;
	uint32_t* output_l1_ptr = (uint32_t*)output_l1_address;

	// Perform Addition
	for(uint32_t i = 0; i < input_size; ++i)
	{
		DPRINT << "input0[" << i << "] " << input0_l1_ptr[i] << ", input1[" << i << "] " << input1_l1_ptr[i] << ENDL();
		output_l1_ptr[i] = input0_l1_ptr[i] + input1_l1_ptr[i];
	}

	// Write output DRAM buffer from L1
	write_dram_pages(output_dram_address, output_l1_address, page_count, page_size);
}