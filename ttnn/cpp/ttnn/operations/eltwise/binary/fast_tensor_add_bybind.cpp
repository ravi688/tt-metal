// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_pybind.hpp"

#include "pybind11/decorators.hpp"
#include "pybind11/export_enum.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
namespace operations {
namespace binary {

namespace detail {

void bind_primitive_binary_operation(py::module& module, const ttnn::prim::fast_tensor_add& operation, const std::string& description) {
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
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
