#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/allocator.hpp>

#include <ostream>
#include <vector>
#include <utility>
#include <chrono>
#include <cassert>


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

struct Result
{
	std::vector<uint32_t> output;
	float enqueue_program_time;
	float kernel_finish_wait_time;
	float output_readback_time;
};

static Result add_uint32_vector(const std::vector<uint32_t>& input0, const std::vector<uint32_t>& input1, const uint32_t page_count) noexcept
{
	const uint32_t input0_buffer_size = input0.size() * sizeof(uint32_t);
	const uint32_t input1_buffer_size = input1.size() * sizeof(uint32_t);
	assert(input0.size() == input1.size());
	assert(input1_buffer_size % page_count == 0);

	// Reserve storage on L1 of core { 0, 0 }
	// Partition the storage into 3 paritions
	// Populate the input0 into partition #0 (host to L1)
	// Populate the input1 into partition #1 (host to L1)
	// Populate the output into partition #2 (host to L1)
	// Read back the partition #2 to host's vector (L1 to host)

	// Create Device
	uint32_t device_id = 0;
	tt::tt_metal::IDevice* device = tt::tt_metal::CreateDevice(device_id);

	// Get Command Queue (id = 0)
	tt::tt_metal::CommandQueue& command_queue = device->command_queue();

	// Create DRAM buffer for input vector 0
	tt::tt_metal::InterleavedBufferConfig dram_buffer_config
	{
		.device = device,
		.size = input0_buffer_size,
		.page_size = input0_buffer_size / page_count
	};
	std::shared_ptr<tt::tt_metal::Buffer> input0_dram_buffer = tt::tt_metal::CreateBuffer(dram_buffer_config);

	// Create DRAM buffer for input vector 1
	std::shared_ptr<tt::tt_metal::Buffer> input1_dram_buffer = tt::tt_metal::CreateBuffer(dram_buffer_config);

	// Populate the DRAM buffers with input vectors
	tt::tt_metal::EnqueueWriteBuffer(command_queue, input0_dram_buffer, const_cast<std::vector<uint32_t>&>(input0), false);
	tt::tt_metal::EnqueueWriteBuffer(command_queue, input1_dram_buffer, const_cast<std::vector<uint32_t>&>(input1), false);

	// Create DRAM buffer for output
	std::shared_ptr<tt::tt_metal::Buffer> output_dram_buffer = tt::tt_metal::CreateBuffer(dram_buffer_config);

	// Just a single core will perform the computation (addition)
	CoreCoord single_core { 0, 0 };

	// Create Program
	tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

	// Create Compute Kernel
	uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
	tt::tt_metal::ComputeConfig data_move_kernel_config { };
	datamove_data_move_config.compile_args = std::vector<uint32_t>
										{ 
											page_count,
											l1_unreserved_base,
											input0.size(),
											input0_dram_buffer->address(),
											input1_dram_buffer->address(),
											output_dram_buffer->address()
										};
	const std::string kernel_file_path = "tt_metal/programming_examples/rpsingh_examples/add_2_integers_in_riscv/kernels/data_move_kernel.cpp";
	[[maybe_unused]] tt::tt_metal::KernelHandle kernel_handle = tt::tt_metal::CreateKernel(program, kernel_file_path, single_core, data_move_kernel_config);

	auto start = std::chrono::steady_clock::now();
	// Launch Program using Fast Dispatch
	tt::tt_metal::EnqueueProgram(command_queue, program, false);
	auto end = std::chrono::steady_clock::now();
	auto enqueue_elapsed = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();

	// While the kernel is being executed, let's allocate output buffer on the host side
	std::vector<uint32_t> output;
	output.reserve(input0.size());

	start = std::chrono::steady_clock::now();
	// Wait for device to complete kernel execution
	tt::tt_metal::Finish(command_queue);
	end = std::chrono::steady_clock::now();
	auto kernel_finish_wait_time = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count();

	start = std::chrono::steady_clock::now();
	// Read back from output DRAM buffer to the host vector
	tt::tt_metal::EnqueueReadBuffer(command_queue, output_dram_buffer, output, true);
	auto output_readback_time = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count(); 

	// Close the device
	tt::tt_metal::CloseDevice(device);

	return { std::move(output), enqueue_elapsed, kernel_finish_wait_time, output_readback_time };
}


int main()
{
	std::vector<uint32_t> input_values0 = { 1, 2, 3, 4, 5, 6 };
	std::vector<uint32_t> input_values1 = { 7, 8, 9, 10, 11, 12 };

	std::cout << "input_values0: " << input_values0 << "\n";
	std::cout << "input_values1: " << input_values1 << "\n";

	// Page Count = 4, so the page size will be 6 bytes as we have only 6 x 4 = 24 bytes of data
	auto [output_values, enqueue_time, kernel_finish_wait_time, output_readback_time] = add_uint32_vector(input_values0, input_values1, 4);

	std::cout << "output_values: " << output_values << "\n";
	std::cout << "enqueue_time: " << enqueue_time << " ms\n";
	std::cout << "kernel_finish_wait_time: " << kernel_finish_wait_time << " ms\n";
	std::cout << "output_readback_time: " << output_readback_time << " ms" << std::endl;
}