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

    uint32_t num_elements = input_tensor_a.get_logical_shape().volume();

    printf("num_elements: %lu", num_elements);

    std::vector<uint32_t> compute_kernel_compute_args = 
    {
        src1_buffer->address(),
        src2_buffer->address(),
        dst_buffer->address(),
        num_elements
    };


    ComputeConfig compute_kernel_config { };
    compute_kernel_config.compile_args = std::move(compute_kernel_compute_args);

    constexpr CoreCoord core = { 0, 0 };
    // Create Compute kernel (Loads the two tiles from L1 and computes addition and writes back to L1)
    tt::tt_metal::KernelHandle compute_kernel_handle = tt::tt_metal::CreateKernel(program,
                                                        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/fast_tensor_add_kernel.cpp",
                                                        core, std::move(compute_kernel_config));

  
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
