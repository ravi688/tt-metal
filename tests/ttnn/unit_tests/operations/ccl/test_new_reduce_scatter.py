# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_pcc
from tests.ttnn.unit_tests.operations.ccl.test_new_all_reduce import check_mesh_tensor_alloc
from models.utility_functions import skip_for_grayskull
from tests.ttnn.unit_tests.operations.ccl.test_all_gather import is_unsupported_case

from ttnn import ShardTensorToMesh, ConcatMeshToTensor


def create_global_semaphores(mesh_device, num_devices, cores, initial_value):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(1)]
    return ccl_semaphore_handles


def run_reduce_scatter_impl(
    t3k_mesh_device,
    num_devices,
    rs_input_shape,
    dim,
    num_links,
    rs_input_dtype,
    layout,
    mem_config_input,
    mem_config_rs,
    rs_topology,
    use_program_cache,
    rs_num_batches,
    num_iters=1,
    enable_trace=True,
):
    torch.manual_seed(0)

    tile = (32, 32)

    devices = t3k_mesh_device.get_devices()

    ##### Fabric setup #####
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    t3k_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphores(t3k_mesh_device, num_devices, ccl_sub_device_crs, 0) for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    single_batch_input_shape = rs_input_shape
    single_batch_input_shape[2] //= rs_num_batches
    persistent_intermediate_buffers = [
        ttnn.from_torch(
            torch.zeros(single_batch_input_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]
    rs_output_shape = rs_input_shape
    rs_output_shape[3] //= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(rs_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=rs_input_dtype,
            memory_config=mem_config_rs,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    for im_buf, out_buf in zip(persistent_intermediate_buffers, persistent_output_buffers):
        check_mesh_tensor_alloc(im_buf)
        check_mesh_tensor_alloc(out_buf)
    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"Reduce scatter shape: {rs_input_shape}")
    logger.info(f"Reduce scatter dim: {dim}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        rs_global_input_shape = rs_input_shape
        rs_global_input_shape[3] *= num_devices
        rs_input_tensor = torch.rand(rs_global_input_shape).bfloat16()
        input_tensors = torch.chunk(rs_input_tensor, num_devices, dim)
        torch_input_tensor_list.append(input_tensors)
        tt_input_tensors = []
        for j, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, rs_input_dtype).to(layout))
        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(t3k_mesh_device, mem_config_input)

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_reduce_scatter_output_list = []
    for i in range(num_iters):
        reduce_output = torch.sum(torch.stack(torch_input_tensor_list[i]), dim=0)
        scatter_output = torch.chunk(reduce_output, num_devices, dim)
        torch_reduce_scatter_output_list.append(scatter_output)

    ##### Perform the TT ops #####
    tt_reduce_scatter_output_list = []

    def run_op(i):
        tt_reduce_scatter_output_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            tt_input_tensor_mesh_list[i],
            persistent_intermediate_buffer=persistent_intermediate_buffers[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            num_batches=rs_num_batches,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config_rs,
            topology=rs_topology,
            subdevice_id=worker_sub_device_id,
        )

        return tt_reduce_scatter_output_tensor

    if enable_trace:
        assert num_iters == 1, "When running in trace, use num_iters = 1"

        # Compile the op
        tt_reduce_scatter_output_tensor = run_op(0)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        tt_reduce_scatter_output_tensor = run_op(0)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)

        tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)
    else:
        for i in range(num_iters):
            tt_reduce_scatter_output_tensor = run_op(i)
            tt_reduce_scatter_output_list.append(tt_reduce_scatter_output_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_rs_out_tensor = tt_reduce_scatter_output_list[i]
        torch_rs_out_tensor = torch_reduce_scatter_output_list[i]

        tt_rs_out = ttnn.from_device(tt_rs_out_tensor)
        tt_rs_out = ttnn.to_torch(tt_rs_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=3))
        eq, output = comp_pcc(tt_rs_out, torch_rs_out_tensor)
        logger.info(f"{output}, iteration {i}")
        assert eq, f"{i} FAILED ag: {output}"

        # print(f"RS TORCH TENSOR {torch_rs_out_tensor}")
        # print(f"RS TT TENSOR {tt_rs_out}")

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


# Enumerate the post-commit cases explicitly
@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize(
    "num_devices, num_links, rs_input_shape, dim, layout, rs_input_dtype, rs_num_batches",
    [
        (
            8,
            1,
            [1, 1, 4096, 2560],
            3,
            ttnn.TILE_LAYOUT,
            ttnn.bfloat16,
            1,
        ),  # Full SD3.5 shape, when reduce scatter unfused
        #        (8, 1, [1, 1, 4096, 2560], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16, 8), # use batching when fused
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_rs",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace",
    [
        # True,
        False,
    ],
)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        # ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
        # (
        #    {"trace_region_size": 90112},
        #    ttnn.Topology.Ring,
        # ),
    ],
    indirect=["device_params"],
)
def test_reduce_scatter_async(
    t3k_mesh_device,
    num_devices,
    num_links,
    rs_input_shape,
    dim,
    layout,
    rs_input_dtype,
    mem_config_input,
    mem_config_rs,
    enable_trace,
    use_program_cache,
    rs_topology,
    rs_num_batches,
):
    run_reduce_scatter_impl(
        t3k_mesh_device,
        num_devices,
        rs_input_shape,
        dim,
        num_links,
        rs_input_dtype,
        layout,
        mem_config_input,
        mem_config_rs,
        use_program_cache=use_program_cache,
        rs_topology=rs_topology,
        enable_trace=enable_trace,
        num_iters=1,
        rs_num_batches=rs_num_batches,
    )
