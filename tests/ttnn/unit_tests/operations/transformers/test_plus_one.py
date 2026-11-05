# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.int32,
        ttnn.uint32,
    ],
)
def test_plus_one(device, w, dtype):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, device=device)
    ttnn.plus_one(input_tensor)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
def test_plus_one_subdevice(device, w):
    torch_input_tensor = torch.randint(32000, (w,))
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))])
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
def test_plus_one_subdevice_nd(device, input_shape):
    torch_input_tensor = torch.randint(32000, input_shape)
    torch_output_tensor = torch_input_tensor + 1
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(
        input_tensor,
        sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1))]),
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [1, 4, 8, 32])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries(device, w, val):
    torch_input_tensor = torch.randint(32000, (w,))
    mask = torch.rand(w) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("input_shape", [(16, 32), (32, 32), (4, 16, 32), (4, 8, 16, 32)])
@pytest.mark.parametrize("val", [-1, -100])
def test_plus_one_with_neg_entries_nd(device, input_shape, val):
    torch_input_tensor = torch.randint(32000, input_shape)
    mask = torch.rand(input_shape) < 0.3
    torch_input_tensor[mask] = val
    torch_output_tensor = torch.where(torch_input_tensor < 0, torch_input_tensor, torch_input_tensor + 1)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device)
    ttnn.plus_one(input_tensor, skip_negative_entries=True)
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("w", [4, 8, 32])
def test_plus_one_program_cache(device, w):
    """Two distinct same-spec DRAM tensors: the second call is a program-cache hit, so the
    cached program must be re-bound to the second tensor's buffer. A stale binding would
    increment the first tensor again and leave the second untouched — validating both
    tensors catches either failure."""
    torch_a = torch.randint(32000, (w,))
    torch_b = torch.randint(32000, (w,))

    tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.int32, device=device)
    tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.int32, device=device)

    ttnn.plus_one(tensor_a)
    ttnn.plus_one(tensor_b)
    assert device.num_program_cache_entries() == 1

    assert_with_pcc(torch_a + 1, ttnn.to_torch(tensor_a), 0.9999)
    assert_with_pcc(torch_b + 1, ttnn.to_torch(tensor_b), 0.9999)


@pytest.mark.parametrize("w", [4, 8, 32])
def test_plus_one_sharded_program_cache(device, w):
    """L1-sharded inputs: the program's working buffer is the input shard itself, so on a
    program-cache hit the framework must re-apply the second tensor's shard address.
    Validating both tensors catches a stale rebinding in either direction."""
    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(shard_grid, (1, w), ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_a = torch.randint(32000, (1, w))
    torch_b = torch.randint(32000, (1, w))

    tensor_a = ttnn.from_torch(torch_a, dtype=ttnn.int32, device=device, memory_config=memory_config)
    tensor_b = ttnn.from_torch(torch_b, dtype=ttnn.int32, device=device, memory_config=memory_config)

    ttnn.plus_one(tensor_a)
    ttnn.plus_one(tensor_b)
    assert device.num_program_cache_entries() == 1

    assert_with_pcc(torch_a + 1, ttnn.to_torch(tensor_a), 0.9999)
    assert_with_pcc(torch_b + 1, ttnn.to_torch(tensor_b), 0.9999)


@pytest.mark.parametrize("input_shape", [(32,), (4, 32), (2, 8, 32)])
def test_plus_one_l1_interleaved(device, input_shape):
    """L1-interleaved input is incremented in place, like DRAM input."""
    torch_input_tensor = torch.randint(32000, input_shape)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.int32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ttnn.plus_one(input_tensor)
    assert torch.equal(ttnn.to_torch(input_tensor), torch_input_tensor + 1)


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_plus_one_multi_core_sub_core_grids(device, memory_config):
    """Interleaved input on a multi-core sub_core_grids is incremented exactly once."""
    torch_input_tensor = torch.randint(32000, (4, 32))
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.int32, device=device, memory_config=memory_config)
    ttnn.plus_one(
        input_tensor, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    )
    assert torch.equal(ttnn.to_torch(input_tensor), torch_input_tensor + 1)


@pytest.mark.parametrize(
    "layout,shape,shard_shape,grid_end",
    [
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (8, 32), (4, 32), (1, 0)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (8, 64), (8, 32), (1, 0)),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (8, 64), (4, 32), (1, 1)),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (8, 1), (8, 16), (0, 0)),
    ],
)
@pytest.mark.parametrize("explicit_grid", [False, True])
@pytest.mark.parametrize("skip_negative_entries", [False, True])
def test_plus_one_multi_row_shards(device, layout, shape, shard_shape, grid_end, explicit_grid, skip_negative_entries):
    offset = int(explicit_grid)
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(offset, offset), ttnn.CoreCoord(grid_end[0] + offset, grid_end[1] + offset))]
    )
    memory_config = ttnn.MemoryConfig(
        layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    x = (torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape) % 41) - 3
    tensor = ttnn.from_torch(
        x, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    expected = torch.where(x < 0, x, x + 1) if skip_negative_entries else x + 1
    ttnn.plus_one(tensor, sub_core_grids=grid if explicit_grid else None, skip_negative_entries=skip_negative_entries)
    assert torch.equal(ttnn.to_torch(tensor), expected)


@pytest.mark.skipif(
    os.environ.get("TT_METAL_ALLOCATOR_MODE_HYBRID") != "1", reason="Requires TT_METAL_ALLOCATOR_MODE_HYBRID=1"
)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize("skip_negative_entries", [False, True])
def test_plus_one_unused_shard_cores(mesh_device, dtype, orientation, skip_negative_entries):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))])
    unused_core = ttnn.CoreCoord(1, 1)
    guard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(unused_core, unused_core)])
    memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [2, 32], orientation),
    )
    memory.experimental_set_range_lockstep_allocation(True)
    guard_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(guard_grid, [2, 32], orientation),
    )
    guard_memory.experimental_set_per_core_allocation(True)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    guard_data = torch.full((2, 32), 123, dtype=torch.int32)
    guard = ttnn.from_torch(
        guard_data,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
        memory_config=guard_memory,
    )
    x = torch.arange(128, dtype=torch.int32).reshape(4, 32)
    if dtype == ttnn.int32:
        x -= 3
    tensor = ttnn.from_torch(
        x,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=mapper,
        memory_config=memory,
    )
    (coord,) = guard.device_coords()
    assert tensor.buffer_address() == guard.experimental_per_core_buffer_address(coord, unused_core)
    for _ in range(2):
        ttnn.plus_one(tensor, skip_negative_entries=skip_negative_entries)
        x = torch.where(x < 0, x, x + 1) if skip_negative_entries else x + 1
        assert torch.equal(ttnn.to_torch(guard, mesh_composer=composer).to(torch.int64), guard_data.to(torch.int64))
        assert torch.equal(ttnn.to_torch(tensor, mesh_composer=composer).to(torch.int64), x.to(torch.int64))
