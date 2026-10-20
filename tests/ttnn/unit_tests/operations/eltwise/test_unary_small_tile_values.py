# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("tile_height", [16, 32])
@pytest.mark.parametrize("dtype,host_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("operation", ["zeros_like", "full_like", "neg"])
def test_unary_small_tile_preserves_neighbour(device, tile_height, dtype, host_dtype, operation):
    host = torch.full((1, 1, 64, 256), 3.0, dtype=host_dtype)
    source = ttnn.from_torch(host, dtype=dtype, tile=ttnn.Tile([tile_height, 32]),
                             layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sentinel = torch.full((1, 1, 256, 1024), 7.0, dtype=torch.bfloat16)
    guard = ttnn.from_torch(sentinel, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Source and guard are consecutive DRAM allocations; the in-place op cannot move the output.
    assert guard.buffer_address() > source.buffer_address()
    if operation == "zeros_like":
        result = ttnn.zeros_like(source, optional_tensor=source)
        expected = torch.zeros_like(host)
    elif operation == "full_like":
        result = ttnn.full_like(source, 1.5, optional_tensor=source)
        expected = torch.full_like(host, 1.5)
    else:
        result = ttnn.neg(source, output_tensor=source)
        expected = -host
    assert result.buffer_address() == source.buffer_address()
    assert torch.equal(ttnn.to_torch(guard), sentinel), "Unary output values landed in the next tensor"
    assert torch.equal(ttnn.to_torch(result), expected)
    # A fresh output must retain the input tile too, including across a cached call.
    for _ in range(2):
        fresh = ttnn.neg(source)
        assert fresh.spec.tile == source.spec.tile
        assert torch.equal(ttnn.to_torch(fresh), -expected)


@pytest.mark.parametrize("tile_height", [16, 32])
@pytest.mark.parametrize("use_nnz", [False, True])
def test_sparse_matmul_small_tile_zero_fill_preserves_neighbour(device, tile_height, use_nnz):
    in0 = ttnn.from_torch(torch.ones((1, 4, 16, 128), dtype=torch.bfloat16),
                          tile=ttnn.Tile([tile_height, 32]), layout=ttnn.TILE_LAYOUT, device=device)
    in1 = ttnn.from_torch(torch.ones((1, 8, 128, 512), dtype=torch.bfloat16),
                          dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.bfloat16).repeat(4).reshape(1, 4, 1, 8)
    sparsity = ttnn.from_torch(mask, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tile = ttnn.Tile([tile_height, 32])
    holder = ttnn.from_torch(torch.zeros((1, 4, 1, 8, 16, 512), dtype=torch.bfloat16),
                             tile=tile, layout=ttnn.TILE_LAYOUT, device=device)
    sentinel = torch.full((1, 1, 256, 1024), 7.0, dtype=torch.bfloat16)
    guard = ttnn.from_torch(sentinel, layout=ttnn.TILE_LAYOUT, device=device)
    address = holder.buffer_address()
    ttnn.deallocate(holder)
    config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 4), in0_block_w=1,
        out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=1,
        per_core_M=1, per_core_N=1, fuse_batch=False, fused_activation=None, mcast_in0=True)
    kwargs = {"nnz": 16} if use_nnz else {}
    result = ttnn.sparse_matmul(in0, in1, sparsity=sparsity, output_tile=tile,
                                program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)
    assert result.buffer_address() == address, "Output must occupy the allocation below the guard"
    assert torch.equal(ttnn.to_torch(guard), sentinel), "Sparse-matmul zero fill changed the next tensor"
    expected = (mask.reshape(1, 4, 1, 8, 1, 1) * 128).expand(1, 4, 1, 8, 16, 512)
    assert torch.equal(ttnn.to_torch(result), expected)
