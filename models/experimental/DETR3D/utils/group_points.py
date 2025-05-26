# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch.autograd import Function
from scipy.stats import pearsonr
from tests.ttnn.utils_for_testing import assert_with_pcc


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, points, idx):
        B, C, N = points.shape
        _, npoints, nsample = idx.shape
        ctx.save_for_backward(idx)
        ctx.n = N

        output = torch.zeros((B, C, npoints, nsample), device=points.device, dtype=points.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(npoints):
                    for j in range(nsample):
                        output[b, c, i, j] = points[b, c, idx[b, i, j]]

        return output


class GroupingOperationFast(Function):
    @staticmethod
    def forward(ctx, points, idx):
        B, C, N = points.shape
        _, npoint, nsample = idx.shape

        idx = idx.to(torch.int64)
        ctx.save_for_backward(idx)
        ctx.n = N

        points_flat = points.view(B * C, N)
        idx_flat = idx.view(B, npoint * nsample)
        idx_expand = idx_flat.unsqueeze(1).expand(-1, C, -1).contiguous().view(B * C, npoint * nsample)
        out_flat = torch.gather(points_flat, 1, idx_expand)
        output = out_flat.view(B, C, npoint, nsample)

        return output


grouping_operation = GroupingOperation.apply
grouping_operation_fast = GroupingOperationFast.apply


@pytest.mark.parametrize(
    "points_shape, idx_shape",
    [
        ((1, 3, 2048), (1, 1024, 32)),
        ((1, 3, 20000), (1, 2048, 64)),
        ((1, 256, 2048), (1, 1024, 32)),
    ],
)
def test_grouping_operation_exact_pcc(points_shape, idx_shape):
    torch.manual_seed(42)
    points = torch.rand(*points_shape, dtype=torch.float32)
    idx = torch.randint(0, points_shape[2], idx_shape, dtype=torch.int64)

    ref_out = grouping_operation(points.clone(), idx.clone()).cpu()
    fast_out = grouping_operation_fast(points.clone(), idx.clone()).cpu()

    # Exact elementwise equality
    assert torch.equal(ref_out, fast_out), f"Outputs differ! Max diff: {(ref_out - fast_out).abs().max().item()}"

    # Pearson Correlation Coefficient (should be exactly 1)
    pcc, _ = pearsonr(ref_out.numpy().flatten(), fast_out.numpy().flatten())
    assert_with_pcc(ref_out.numpy().flatten(), fast_out.numpy().flatten(), 1.0)
    assert pcc == 1.0, f"PCC != 1.0, got {pcc}"
