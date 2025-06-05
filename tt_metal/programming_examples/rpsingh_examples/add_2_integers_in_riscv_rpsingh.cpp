#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>

#include <ostream>
#include <vector>


template<typename T>
static std::ostream& operator <<(std::ostream& stream, const std::vector<T>& v) noexcept
{
	stream << "{ ";
	for(std::size_t i = 0; i < v.size(); ++i)
	{
		stream << v[i];
		if((i + 1) != v.size())
			stream << ", ";
	}
	stream << " }";
	return stream;
}

static std::vector<uint32_t> add_uint32_vector(const std::vector<uint32_t>& input0, const std::vector<uint32_t>& input1) noexcept
{
	// Reserve storage on L1 of core { 0, 0 }
	// Partition the storage into 3 paritions
	// Populate the input0 into partition #0 (host to L1)
	// Populate the input1 into partition #1 (host to L1)
	// Populate the output into partition #2 (host to L1)
	// Read back the partition #2 to host's vector (L1 to host)

	// Create Device
	uint32_t device_id = 0;
	tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(device_id);

	// Create the base address in L1
	uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

	// Just a single core will perform the computation (addition)
	CoreCoord single_core { 0, 0 };
	
	auto partition0_base_addr = l1_unreserved_base;
	// Write the first input vector to the first partition #0
	tt::tt_metal::detail::WriteToDeviceL1(device, single_core, partition0_base_addr, input0);
	// Write the second input vector to the second partition #1
	auto partition1_base_addr = partition0_base_addr + sizeof(uint32_t) * input0.size();
	tt::tt_metal::detail::WriteToDeviceL1(device, single_core, partition1_base_addr, input1);

	// Create Program
	tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

	// Create Compute Kernel
	auto partition2_base_addr = partition1_base_addr + sizeof(uint32_t) * input1.size();
	tt::tt_metal::ComputeConfig compute_kernel_config { };
	compute_kernel_config.compile_args = std::vector<uint32_t>
										{ 
											input0.size(),
											partition0_base_addr,
											partition1_base_addr,
											partition2_base_addr
										};
	const std::string kernel_file_path = "tt_metal/programming_examples/rpsingh_examples/kernels/add_2_integers_in_riscv_kernel_rpsingh.cpp";
	[[maybe_unused]] tt::tt_metal::KernelHandle kernel_handle = tt::tt_metal::CreateKernel(program, kernel_file_path, single_core, compute_kernel_config);

	// Launch Program using Slow Dispatch (for fast dispatch we could have used CommandQueue and EnqueuPreogram())
	tt::tt_metal::detail::LaunchProgram(device, program);

	// Readback output (from partition #2) to host
	std::vector<uint32_t> output;
	output.reserve(input0.size());
	tt::tt_metal::detail::ReadFromDeviceL1(device, single_core, partition2_base_addr, sizeof(uint32_t) * input0.size(), output);

	// Close the device
	tt::tt_metal::CloseDevice(device);

	return output;
}


int main()
{
	std::vector<uint32_t> input_values0 = { 1, 2, 3, 4, 5, 6 };
	std::vector<uint32_t> input_values1 = { 7, 8, 9, 10, 11, 12 };

	std::cout << "input_values0: " << input_values << "\n";
	std::cout << "input_values1: " << input_values << "\n";

	std::vector<uint32_t> output_values = add_uint32_vector(input_values0, input_values1);

	std::cout << "output_values: " << output_values << std::endl;
}