# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from scipy.stats import pearsonr
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_pcc
import time


# ---------------- Reference (Slow) ---------------- #
def furthest_point_sampling(points, n_samples):
    B, N, _ = points.shape
    idxs = torch.zeros((B, n_samples), dtype=torch.int32, device=points.device)
    temp = torch.full((B, N), 1e10, dtype=torch.float32, device=points.device)

    for b in range(B):
        old = 0
        idxs[b, 0] = old

        for j in range(1, n_samples):
            best_dist = -1
            best_idx = 0
            x1, y1, z1 = points[b, old, :]

            for k in range(N):
                x2, y2, z2 = points[b, k, :]
                dist = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
                dist = torch.min(dist, temp[b, k])
                temp[b, k] = dist
                if dist > best_dist:
                    best_dist = dist
                    best_idx = k

            idxs[b, j] = best_idx
            old = best_idx

    return idxs


# ---------------- Fast Vectorized ---------------- #
def furthest_point_sampling_fast(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    B, N, _ = points.shape
    device = points.device
    centroids = torch.zeros((B, n_samples), dtype=torch.long, device=device)
    distance = torch.ones((B, N), dtype=points.dtype, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=2)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=1)[1]

    return centroids


# ---------------- Test Cases ---------------- #


@pytest.mark.parametrize(
    "shape,n_samples",
    [
        ((1, 20000, 3), 2048),
        ((1, 2048, 3), 1024),
        ((1, 1024, 3), 128),
        # for 2048 sliced input tensor
        # ((1, 2048, 3), 2048),
        # ((1, 2048, 3), 1024),
        # ((1, 1024, 3), 128),
    ],
)
def test_fps_matches_exactly(shape, n_samples):
    torch.manual_seed(0)
    points = torch.rand(*shape)

    # Time the reference (slow) implementation
    start_ref = time.time()
    ref_idx = furthest_point_sampling(points.clone(), n_samples).cpu().numpy().flatten()
    end_ref = time.time()
    ref_time = end_ref - start_ref

    # Time the fast implementation
    start_fast = time.time()
    fast_idx = furthest_point_sampling_fast(points.clone(), n_samples).cpu().numpy().flatten()
    end_fast = time.time()
    fast_time = end_fast - start_fast

    # Check correctness
    assert ref_idx.shape == fast_idx.shape, "Output shape mismatch"
    assert np.array_equal(ref_idx, fast_idx), f"Mismatch in selected indices!"
    pcc, _ = pearsonr(ref_idx.astype(np.float32), fast_idx.astype(np.float32))
    pcc2 = assert_with_pcc(ref_idx, fast_idx, 1.0)
    assert pcc == 1.0, f"PCC != 1.0 for shape {shape}, got {pcc}"

    # Print execution times
    print(f"\nShape: {shape}, Samples: {n_samples}")
    print(f"Reference (slow) time: {ref_time:.6f} seconds")
    print(f"Fast (vectorized) time: {fast_time:.6f} seconds")
    print(f"Speedup: {ref_time / fast_time:.2f}x\n")
