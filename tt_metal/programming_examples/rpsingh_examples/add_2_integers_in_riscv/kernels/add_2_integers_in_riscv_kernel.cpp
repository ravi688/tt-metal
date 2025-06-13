
#include "compute_kernel_api.h"

#include <cstdint>

using namespace std;


// Compute kernels must have their entry point wrapped inside namespace 'NAMESPACE'
namespace NAMESPACE
{

// Compute kernels must have their entry point name as 'MAIN' without parentheses
void MAIN
{
	uint32_t num_elements = get_compile_time_arg_val(0);
	uint32_t partition0_base_addr = get_compile_time_arg_val(1);
	uint32_t partition1_base_addr = get_compile_time_arg_val(2);
	uint32_t partition2_base_addr = get_compile_time_arg_val(3);


	const uint32_t* part0_elements = (uint32_t*)partition0_base_addr;
	const uint32_t* part1_elements = (uint32_t*)partition1_base_addr;
	uint32_t* part2_elements = (uint32_t*)partition2_base_addr;

	for(uint32_t i = 0; i < num_elements; ++i)
		part2_elements[i] = part0_elements[i] + part1_elements[i];

}

}