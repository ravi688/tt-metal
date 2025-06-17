// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fast_tensor_add_device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::binary {
FastTensorAddDeviceOperation::SingleCore::cached_program_t FastTensorAddDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    // LOGIC for single core only
    // 1. A core's SRAM can store a fixed size data
    // 2. Reader Kernel: If tensors are big then they need to be partitioned into smaller chunks (tiles) which be placed into the core's SRAM one at a time
    // 3. Compute Kernel: For a pair of input tiles, addition will be performed in the RISC-V core
    // 4. Compute Kernel: The RISC-V core would write to the SRAM after each addition
    // 5. Writer Kernel: The result tiles wopuld be assembled into the DRAM of the device

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src1_buffer = input_tensor_a.buffer();
    auto src2_buffer = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();


    // Create a program
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& tensor_shape = input_tensor_a.logical_shape();

    std::cout << "Tensor Shape: " << tensor_shape[0] << " x " << tensor_shape[1] << std::endl;

    std::vector<uint32_t> compute_kernel_compute_args = 
    {
        src1_buffer->address(),
        src2_buffer->address(),
        dst_buffer->address(),
        tensor_shape[1]
    };

    ComputeConfig compute_kernel_config { };
    compute_kernel_config.compile_args = std::move(compute_kernel_compute_args);

    const uint32_t num_rows = tensor_shape[0];
    assert(num_rows >= 1);
    const CoreRange core_range { { 0, 0 }, { num_rows - 1, 0 } };
    // Create Reader kernel (loads data from 2 L1 pointers and writes to 2 circular buffers)
    tt::tt_metal::KernelHandle reader_kernel_handle = tt::tt_metal::CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/fast_tensor_add_reader_kernel.cpp",
                                                        core_range, std::move(reader_kernel_config));
    // Create Writer kernel (reads data from 1 circular buffer and writes to 1 L1 pointer)
    tt::tt_metal::KernelHandle writer_kernel_handle = tt::tt_metal::CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/fast_tensor_add_writer_kernel.cpp",
                                                        core_range, std::move(writer_kernel_config));
    // Create Compute kernel (reads data from 2 circular buffers, computes, and writes the result to 1 circular buffer)
    tt::tt_metal::KernelHandle compute_kernel_handle = tt::tt_metal::CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/fast_tensor_add_kernel.cpp",
                                                        core_range, std::move(compute_kernel_config));

    // Circular buffer configs
    tt::tt_metal::CircularBufferConfig cb_config1 { 32 * 32 * 4, { 0, tt::DataFormat::Float32 } };
    cb_config.set_page_size(0, 32 * 32 * 4);
    tt::tt_metal::CircularBufferConfig cb_config2 { 32 * 32 * 4, { 1, tt::DataFormat::Float32 } };
    cb_config.set_page_size(1, 32 * 32 * 4);
    tt::tt_metal::CircularBufferConfig cb_config3 { 32 * 32 * 4, { 2, tt::DataFormat::Float32 } };
    cb_config.set_page_size(2, 32 * 32 * 4);

    std::array<uint32_t, 5> reader_kernel_args = { 0, 1, 2,  src1_buffer->address(), src2_buffer->address() };
    std::array<uint32_t, 2> writer_kernel_args = { 0, dst_buffer->address() };
    std::array<uint32_t, 3> compute_kernel_args = { 0, 1, 2 };
    
    for(uint32_t i = 0; i < num_rows; ++i)
    {
        CoreCoord core { i, 0 }; 
        // The reader kernel will write to these circular buffers copying from L1
        // The compute kernel will read from these circular buffers
        [[maybe_unused]] tt::tt_metal::CBHandle input0_cb_handle = tt::tt_metal::CreateCircularBuffer(program, core, cb_config1);
        [[maybe_unused]] tt::tt_metal::CBHandle input1_cb_handle = tt::tt_metal::CreateCircularBuffer(program, core, cb_config2);

        // The compute kernel will write to this circular buffer (addition result)
        // The writer kernel will read from this circular buffer writing to L1
        [[maybe_unused]] tt::tt_metal::CBHandle output_cb_handle = tt::tt_metal::CreateCircularBuffer(program, core, cb_config3);

        // Set runtime arguments
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_handle, core, reader_kernel_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_handle, core, writer_kernel_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_handle, core, compute_kernel_args);
    }

    return { std::move(program), { .compute_kernel_id = compute_kernel_handle }};
}

void FastTensorAddDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // auto& program = cached_program.program;
    // auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;

    // const auto& input_tensor_a = tensor_args.input_tensor_a;
    // const auto& input_tensor_b = tensor_args.input_tensor_b;
    // auto& output_tensor = tensor_return_value;

    // auto src1_buffer = input_tensor_a.buffer();
    // auto src2_buffer = input_tensor_b.buffer();
    // auto dst_buffer = output_tensor.buffer();
    
    // auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, compute_kernel_id, CoreCoord{0, 0});
    // runtime_args[0] = src1_buffer->address();
    // runtime_args[1] = src2_buffer->address();
    // runtime_args[2] = dst_buffer->address();
    // runtime_args[3] = input_tensor_a.volume();
}

}  // namespace ttnn::operations::binary
