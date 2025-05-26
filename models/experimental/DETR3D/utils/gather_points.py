# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from scipy.stats import pearsonr
import time

from tests.ttnn.utils_for_testing import assert_with_pcc


# ---------------- Reference (Slow) ---------------- #
def gather_points(points, idx):
    B, C, N = points.shape
    M = idx.shape[1]
    out = torch.zeros((B, C, M), dtype=points.dtype, device=points.device)

    for i in range(B):
        for l in range(C):
            for j in range(M):
                out[i, l, j] = points[i, l, idx[i, j]]

    return out


# ---------------- Fast Vectorized ---------------- #
def gather_points_fast(points, idx):
    B, C, N = points.shape
    M = idx.shape[1]
    idx_expand = idx.long().unsqueeze(1).expand(B, C, M)
    out = torch.gather(points, 2, idx_expand)
    return out


# ---------------- Test Cases ---------------- #
@pytest.mark.parametrize(
    "B,C,N,M",
    [
        (1, 3, 20000, 2048),
        (1, 3, 2048, 1024),
        # (2, 5, 4096, 2048),
        # (1, 1, 1000, 500),
    ],
)
def test_gather_points_equivalence(B, C, N, M):
    torch.manual_seed(42)
    points = torch.randn(B, C, N, dtype=torch.float32)
    idx = torch.randint(0, N, (B, M), dtype=torch.int32)

    # Time the reference slow implementation
    start_ref = time.time()
    out_ref = gather_points(points, idx)
    end_ref = time.time()
    ref_time = end_ref - start_ref

    # Time the fast vectorized implementation
    start_fast = time.time()
    out_fast = gather_points_fast(points, idx)
    end_fast = time.time()
    fast_time = end_fast - start_fast

    # Check output shapes
    assert out_ref.shape == out_fast.shape, "Output shapes do not match"

    # Check exact equality (float32 should be exactly same here)
    assert torch.allclose(out_ref, out_fast, atol=1e-6), "Outputs differ!"

    # Flatten outputs for PCC check
    out_ref_np = out_ref.cpu().numpy().flatten()
    out_fast_np = out_fast.cpu().numpy().flatten()

    # Compute Pearson Correlation Coefficient
    pcc, _ = pearsonr(out_ref_np, out_fast_np)
    assert_with_pcc(out_ref_np, out_fast_np, 1.0)
    assert pcc == 1.0, f"PCC is not 1.0 but {pcc}"

    # Print timing results
    print(f"\nB={B}, C={C}, N={N}, M={M}")
    print(f"Slow implementation time: {ref_time:.6f} sec")
    print(f"Fast implementation time: {fast_time:.6f} sec")
    print(f"Speedup: {ref_time/fast_time:.2f}x\n")
