# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from scipy.stats import pearsonr
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_pcc
import time


def query_ball_point(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    assert new_xyz.is_contiguous()
    assert xyz.is_contiguous()
    assert new_xyz.dtype == torch.float32
    assert xyz.dtype == torch.float32

    if new_xyz.is_cuda:
        assert xyz.is_cuda

    b, m, _ = new_xyz.size()
    _, n, _ = xyz.size()

    idx = torch.zeros((b, m, nsample), dtype=torch.int, device=new_xyz.device)

    radius2 = radius * radius

    for batch_index in range(b):
        new_xyz_batch = new_xyz[batch_index]
        xyz_batch = xyz[batch_index]
        idx_batch = idx[batch_index]

        for j in range(m):
            new_x, new_y, new_z = new_xyz_batch[j]
            distances = torch.sum((xyz_batch - new_xyz_batch[j]) ** 2, dim=1)
            mask = distances < radius2
            valid_indices = torch.nonzero(mask, as_tuple=True)[0]

            if valid_indices.numel() > 0:
                count = min(nsample, valid_indices.numel())
                idx_batch[j, :count] = valid_indices[:count]
                if count < nsample:
                    idx_batch[j, count:] = valid_indices[0]

    return idx


def query_ball_point_fast_exact(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    b, m, _ = new_xyz.shape
    _, n, _ = xyz.shape
    device = new_xyz.device
    radius2 = radius * radius

    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist2 = torch.sum(diff**2, dim=3)
    mask = dist2 < radius2

    idx = torch.zeros((b, m, nsample), dtype=torch.int32, device=device)

    arange_n = torch.arange(n, device=device).view(1, 1, n).expand(b, m, n)
    arange_n_masked = torch.where(mask, arange_n, torch.full_like(arange_n, n + 1))

    sorted_indices, _ = torch.sort(arange_n_masked, dim=2)
    first_nsample = sorted_indices[:, :, :nsample]

    invalid_mask = first_nsample == (n + 1)
    first_valid = first_nsample[:, :, 0].unsqueeze(2).expand_as(first_nsample)
    first_nsample[invalid_mask] = first_valid[invalid_mask]

    return first_nsample.to(torch.int32)


@pytest.mark.parametrize(
    "B,M,N,nsample,radius",
    [
        (1, 2048, 20000, 64, 0.2),
        (1, 1024, 2048, 32, 0.4),
        # (1, 256, 1024, 16),
    ],
)
def test_query_ball_point_exact_match(B, M, N, nsample, radius):
    torch.manual_seed(0)
    # radius = 0.2
    new_xyz = torch.rand(B, M, 3, dtype=torch.float32)
    xyz = torch.rand(B, N, 3, dtype=torch.float32)

    start_ref = time.time()
    ref_idx = query_ball_point(new_xyz.clone(), xyz.clone(), radius, nsample).cpu().numpy().flatten()
    end_ref = time.time()

    start_fast = time.time()
    fast_idx = query_ball_point_fast_exact(new_xyz.clone(), xyz.clone(), radius, nsample).cpu().numpy().flatten()
    end_fast = time.time()

    assert ref_idx.shape == fast_idx.shape
    assert np.array_equal(ref_idx, fast_idx), "Fast output does not exactly match reference"

    pcc, _ = pearsonr(ref_idx.astype(np.float32), fast_idx.astype(np.float32))
    assert_with_pcc(ref_idx, fast_idx, 1.0)
    assert pcc == 1.0, f"PCC is not 1.0, got {pcc}"

    print(f"\nTest passed for B={B}, M={M}, N={N}, nsample={nsample}")
    print(f"Reference time: {end_ref - start_ref:.6f}s")
    print(f"Fast time: {end_fast - start_fast:.6f}s")
    print(f"Speedup: {(end_ref - start_ref) / (end_fast - start_fast):.2f}x")
