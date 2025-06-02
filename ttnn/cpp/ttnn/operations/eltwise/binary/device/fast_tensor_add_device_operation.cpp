// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fast_tensor_add_device_operation.hpp"

namespace ttnn::operations::binary {

FastTensorAddDeviceOperation::program_factory_t FastTensorAddDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    bool some_condition_based_on_operation_attributes_and_or_tensor_args = true;
    if (some_condition_based_on_operation_attributes_and_or_tensor_args) {
        return SingleCore{};
    }
    return MultiCore{};
}

void FastTensorAddDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

void FastTensorAddDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

FastTensorAddDeviceOperation::spec_return_value_t FastTensorAddDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return TensorSpec(
        input_tensor.get_logical_shape(),
        tt::tt_metal::TensorLayout(
            input_tensor.get_dtype(), tt::tt_metal::PageConfig(input_tensor.get_layout()), MemoryConfig{}));
}

FastTensorAddDeviceOperation::tensor_return_value_t FastTensorAddDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensor.device());
}

std::tuple<FastTensorAddDeviceOperation::operation_attributes_t, FastTensorAddDeviceOperation::tensor_args_t>
FastTensorAddDeviceOperation::invoke(const Tensor& input_tensor) {
    return {operation_attributes_t{true, 42}, tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::binary
