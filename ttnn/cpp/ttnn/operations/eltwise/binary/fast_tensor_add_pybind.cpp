// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "pybind11/export_enum.hpp"
#include "ttnn/operations/eltwise/binary/device/fast_tensor_add_device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace binary {

namespace detail {

void bind_primitive_fast_tensor_add_operation(py::module& module, const ttnn::prim::fast_tensor_add& operation, const std::string& description) {
    auto doc = std::string {
        R"doc(
        Fast Tensor Addition

        Args:
            * :attr:`input_tensor_a` (ttnn.Tensor)
            * :attr:`input_tensor_b` (ttnn.Tensor): the tensor add to :attr:`input_tensor_a`.

        Example:

            >>> tensor1 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> tensor2 = ttnn.to_device(ttnn.from_torch(torch.tensor(([[1, 2], [3, 4]]), dtype=torch.bfloat16)), device)
            >>> output = ttnn.fast_tensor_add(tensor1, tensor2)
        )doc" };

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ttnn::prim::fast_tensor_add& self, const ttnn::Tensor& input_tensor_a, const ttnn::Tensor& input_tensor_b) -> ttnn::Tensor {
                return self(input_tensor_a, input_tensor_b);
            },
            py::arg("input_tensor_a"),
            py::arg("input_tensor_b")});
}

}

void py_module(py::module& module) {

    detail::bind_primitive_fast_tensor_add_operation(
        module,
        ttnn::fast_tensor_add,
        R"doc(Adds :attr:`input_tensor_a` to :attr:`input_tensor_b` and returns the tensor with the same layout as :attr:`input_tensor_a`)doc",
        R"doc(\mathrm{{output\_tensor}}_i = \mathrm{{input\_tensor\_a}}_i + \mathrm{{input\_tensor\_b}}_i)doc",
        R"doc(: :code:`'None'` | :code:`'relu'`. )doc",
        R"doc(BFLOAT16, BFLOAT8_B, INT32)doc");

}

}  // namespace binary
}  // namespace operations
}  // namespace ttnn
