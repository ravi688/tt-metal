# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch


def gather_points(points, idx):
    B, C, N = points.shape
    M = idx.shape[1]
    out = torch.zeros((B, C, M), dtype=points.dtype, device=points.device)

    for i in range(B):
        for l in range(C):
            for j in range(M):
                out[i, l, j] = points[i, l, idx[i, j]]

    return out


def gather_points_grad(grad_out, idx, n):
    B, C, M = grad_out.shape
    grad_points = torch.zeros((B, C, n), dtype=grad_out.dtype, device=grad_out.device)

    for i in range(B):
        for l in range(C):
            for j in range(M):
                grad_points[i, l, idx[i, j]] += grad_out[i, l, j]

    return grad_points


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


def three_nn_python(unknown, known):
    B, n, _ = unknown.size()
    m = known.size(1)

    dist2 = torch.zeros(B, n, 3, dtype=unknown.dtype, device=unknown.device)
    idx = torch.zeros(B, n, 3, dtype=torch.int, device=unknown.device)

    for batch_index in range(B):
        for j in range(n):
            ux, uy, uz = unknown[batch_index, j]

            best1, best2, best3 = float("inf"), float("inf"), float("inf")
            besti1, besti2, besti3 = 0, 0, 0

            for k in range(m):
                x, y, z = known[batch_index, k]
                d = (ux - x) ** 2 + (uy - y) ** 2 + (uz - z) ** 2

                if d < best1:
                    best3, best2, best1 = best2, best1, d
                    besti3, besti2, besti1 = besti2, besti1, k
                elif d < best2:
                    best3, best2 = best2, d
                    besti3, besti2 = besti2, k
                elif d < best3:
                    best3 = d
                    besti3 = k

            dist2[batch_index, j, 0] = best1
            dist2[batch_index, j, 1] = best2
            dist2[batch_index, j, 2] = best3

            idx[batch_index, j, 0] = besti1
            idx[batch_index, j, 1] = besti2
            idx[batch_index, j, 2] = besti3

    return dist2, idx


def three_interpolate_python(features, idx, weight):
    B, c, m = features.size()
    n = idx.size(1)

    output = torch.zeros(B, c, n, dtype=features.dtype, device=features.device)

    for batch_index in range(B):
        for i in range(n):
            i1, i2, i3 = idx[batch_index, i]
            w1, w2, w3 = weight[batch_index, i]

            output[batch_index, :, i] = (
                features[batch_index, :, i1] * w1
                + features[batch_index, :, i2] * w2
                + features[batch_index, :, i3] * w3
            )

    return output
