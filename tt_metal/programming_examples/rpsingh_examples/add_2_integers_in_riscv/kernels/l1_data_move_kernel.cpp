
#include "dataflow_api.h"
#include "debug/dprint.h"

// On wormhole chip we have 6 DRAM banks
static constexpr uint32_t DRAM_BANK_COUNT = 6;

void read_l1_pages(uint32_t src_l1_address, uint32_t l1_addr, uint32_t page_count, uint32_t page_size)
{
	for(uint32_t i = 0; i < page_count; ++i)
	{
		// Round-robin the bank id around 'DRAM_BANK_COUNT' banks
		uint32_t bank_id = i % DRAM_BANK_COUNT;
		auto l1_noc_addr = get_noc_addr_from_bank_id<false>(bank_id, src_l1_address);
		// Reads should be aligned to 32 bytes on wormhole
		// That means read size should always be a multiple of 32 bytes
		noc_async_read(l1_noc_addr, l1_addr + i * page_size, page_size);
	}
	noc_async_read_barrier();
}

void write_l1_pages(uint32_t dst_l1_address, uint32_t l1_addr, uint32_t page_count, uint32_t page_size)
{
	for(uint32_t i = 0; i < page_count; ++i)
	{
		// Round-robin the bank id around 'DRAM_BANK_COUNT' banks
		uint32_t bank_id = i % DRAM_BANK_COUNT;
		auto l1_noc_addr = get_noc_addr_from_bank_id<false>(bank_id, dst_l1_address);
		// Writes should be aligned to 16 bytes on wormhole
		// That means write size should always be a multiple of 32 bytes
		noc_async_write(l1_addr + i * page_size, l1_noc_addr, page_size);
	}
	noc_async_write_barrier();
}

void kernel_main()
{
	// Number of pages in which the inputs and outputs are divided
	uint32_t page_count = get_compile_time_arg_val(0);
	// L1 base address
	uint32_t l1_mem_addr = get_compile_time_arg_val(1);
	// element counts in the input vectors
	uint32_t input_size = get_compile_time_arg_val(2);
	uint32_t input0_l1_address = get_compile_time_arg_val(3);
	uint32_t input1_l1_address = get_compile_time_arg_val(4);
	uint32_t output_l1_address = get_compile_time_arg_val(5);


	uint32_t input_size_in_bytes = input_size * sizeof(uint32_t);
	
	// Calculate L1 addresses of input vectors and input vectors
	uint32_t input0_l1_buf_address = l1_mem_addr;
	uint32_t input1_l1_buf_address = l1_mem_addr + input_size_in_bytes;
	uint32_t output_l1_buf_address = input1_l1_buf_address + input_size_in_bytes;

	uint32_t page_size = input_size_in_bytes / page_count;

	// Read input DRAM buffers into L1
	// NOTE: DRAM is not directly accessible by any of tensix cores (baby RISC-V cores)
	// It should only be done via NoC read and write requests
	read_l1_pages(input0_l1_address, input0_l1_buf_address, page_count, page_size);
	read_l1_pages(input1_l1_address, input1_l1_buf_address, page_count, page_size);

	// L1 Pointers are directly accessible to Baby RISC-V cores
	uint32_t* input0_l1_ptr = (uint32_t*)input0_l1_buf_address;
	uint32_t* input1_l1_ptr = (uint32_t*)input1_l1_buf_address;
	uint32_t* output_l1_ptr = (uint32_t*)output_l1_buf_address;

	// Perform Addition
	for(uint32_t i = 0; i < input_size; ++i)
		output_l1_ptr[i] = input0_l1_ptr[i] + input1_l1_ptr[i];

	// Write output DRAM buffer from L1
	write_l1_pages(output_l1_address, output_l1_buf_address, page_count, page_size);
}