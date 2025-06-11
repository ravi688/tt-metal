#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/allocator.hpp>

#include <ostream>
#include <vector>
#include <utility>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <random>


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

	constexpr uint32_t cb_page_size = 8 * sizeof(uint32_t);

	assert((input0.size() * sizeof(uint32_t)) % cb_page_size == 0);
	
	const uint32_t cb_page_count = 1;

	// Create circular buffer for feeding input (0) data to the compute core
	// The reader kernel would populate the input data from Device-Local-DRAM buffer to this circular buffer
	// The compute kernel would use/read the input data from this circular buffer
	constexpr uint32_t input0_cb_index = tt::CBIndex::c_0;
	constexpr uint32_t input0_cb_buffer_size = cb_page_count * cb_page_size;
	tt::tt_metal::CircularBufferConfig input0_cb_config(input0_cb_buffer_size, { { input0_cb_index, tt::DataFormat::UInt32 } });
	input0_cb_config.set_page_size(input0_cb_index, cb_page_size);
	tt::tt_metal::CBHandle input0_cb = tt::tt_metal::CreateCircularBuffer(program, single_core, input0_cb_config);

	// Create circular buffer for feeding input (1) data to the compute core
	// The reader kernel would populate the input data from Device-Local-DRAM buffer to this circular buffer
	// The compute kernel would use/read the input data from this circular buffer
	constexpr uint32_t input1_cb_index = tt::CBIndex::c_1;
	constexpr uint32_t input1_cb_buffer_size = cb_page_count * cb_page_size;
	tt::tt_metal::CircularBufferConfig input1_cb_config(input1_cb_buffer_size, { { input1_cb_index, tt::DataFormat::UInt32 } });
	input1_cb_config.set_page_size(input1_cb_index, cb_page_size);
	tt::tt_metal::CBHandle input1_cb = tt::tt_metal::CreateCircularBuffer(program, single_core, input1_cb_config);

	// Create circular buffer for collecting output data from the compute core
	// The writer kernel would read output data from this circular buffer and write to device-local-DRAM buffer
	// The compute kernel would write the output data to this circular buffer
	constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
	constexpr uint32_t output_cb_buffer_size = cb_page_count * cb_page_size;
	tt::tt_metal::CircularBufferConfig output_cb_config(output_cb_buffer_size, { { output_cb_index, tt::DataFormat::UInt32 } });
	output_cb_config.set_page_size(output_cb_index, cb_page_size);
	tt::tt_metal::CBHandle output_cb = tt::tt_metal::CreateCircularBuffer(program, single_core, output_cb_config);

	// Instantiate Kernels

	const std::string compute_kernel_file_path = "tt_metal/programming_examples/rpsingh_examples/add_2_integers_in_riscv/kernels/compute_kernel.cpp";
	const std::string reader_kernel_file_path = "tt_metal/programming_examples/rpsingh_examples/add_2_integers_in_riscv/kernels/reader_kernel.cpp";
	const std::string writer_kernel_file_path = "tt_metal/programming_examples/rpsingh_examples/add_2_integers_in_riscv/kernels/writer_kernel.cpp";

	// Create Compute Kernel
	tt::tt_metal::ComputeConfig compute_kernel_config { };
	{
		std::vector<uint32_t> compile_kernel_compile_args = 
		{
			input0.size(),
			cb_page_size,
			input0_cb_index,
			input1_cb_index,
			output_cb_index
		};
		compute_kernel_config.compile_args = std::move(compile_kernel_compile_args);
	}
	[[maybe_unused]] tt::tt_metal::KernelHandle compute_kernel_handle = tt::tt_metal::CreateKernel(program, compute_kernel_file_path, single_core, compute_kernel_config);

	// Create Reader Kernel
	tt::tt_metal::ReaderDataMovementConfig reader_kernel_config { };
	{
		std::vector<uint32_t> reader_kernel_compile_args = 
		{
			input0.size(),
			page_count,
			input0_cb_index,
			input1_cb_index,
			input0_dram_buffer->address(),
			input1_dram_buffer->address()
		};
		reader_kernel_config.compile_args = std::move(reader_kernel_compile_args);
	}
	[[maybe_unused]] tt::tt_metal::KernelHandle reader_kernel_handle = tt::tt_metal::CreateKernel(program, reader_kernel_file_path, single_core, reader_kernel_config);
	
	// Create Writer Kernel
	tt::tt_metal::WriterDataMovementConfig writer_kernel_config { };
	{
		std::vector<uint32_t> writer_kernel_compile_args = 
		{
			input0.size(),
			page_count,
			output_cb_index,
			output_dram_buffer->address()
		};
		writer_kernel_config.compile_args = std::move(writer_kernel_compile_args);
	}
	[[maybe_unused]] tt::tt_metal::KernelHandle writer_kernel_handle = tt::tt_metal::CreateKernel(program, writer_kernel_file_path, single_core, writer_kernel_config);
	

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
	end = std::chrono::steady_clock::now();
	auto output_readback_time = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end - start).count(); 

	// Close the device
	tt::tt_metal::CloseDevice(device);

	return { std::move(output), enqueue_elapsed, kernel_finish_wait_time, output_readback_time };
}

std::vector<uint32_t> get_populated_vector(uint32_t count)
{
	std::vector<uint32_t> v(count);
	std::ranges::generate(v, []()
	{
		static std::uniform_int_distribution<uint32_t> distribution(1, 10);
		static std::random_device rd;
		static std::mt19937 prng(rd());
		return distribution(prng);
	});
	return v;
}


int main()
{
	std::vector<uint32_t> input_values0 = get_populated_vector(32);
	std::vector<uint32_t> input_values1 = get_populated_vector(32);

	std::cout << "input_values0: " << input_values0 << "\n";
	std::cout << "input_values1: " << input_values1 << "\n";

	// Page Count = 4, so the page size will be 6 bytes as we have only 6 x 4 = 24 bytes of data
	auto [output_values, enqueue_time, kernel_finish_wait_time, output_readback_time] = add_uint32_vector(input_values0, input_values1, 4);

	std::cout << "output_values: " << output_values << "\n";
	std::cout << "enqueue_time: " << enqueue_time << " ms\n";
	std::cout << "kernel_finish_wait_time: " << kernel_finish_wait_time << " ms\n";
	std::cout << "output_readback_time: " << output_readback_time << " ms" << std::endl;
}