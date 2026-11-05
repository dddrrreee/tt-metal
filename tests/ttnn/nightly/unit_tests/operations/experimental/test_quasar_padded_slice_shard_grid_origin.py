# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


def memory(kind, start, size, shard, col=False):
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(start[0] + size[0] - 1, start[1] + size[1] - 1))}
    )
    return ttnn.MemoryConfig(
        getattr(ttnn.TensorMemoryLayout, kind),
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR if col else ttnn.ShardOrientation.ROW_MAJOR),
    )


def put(device, x, mem=ttnn.DRAM_MEMORY_CONFIG, rm=False):
    return ttnn.from_torch(
        x,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )


def compare(actual, expected):
    got = (actual if isinstance(actual, torch.Tensor) else ttnn.to_torch(actual)).float()
    expected = expected.float()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    wrong = got != expected
    return dict(wrong=int(wrong.sum()), elements=got.numel(), max_abs=float((got - expected).abs().max()))


def values(shape, op, start, high=100, salt=0):
    # Distinct placement data prevents stale L1 contents from hiding misplaced reads.
    seed = 100000 * start[0] + 10000 * start[1] + 100 * salt + sum((i + 1) * ord(c) for i, c in enumerate(op))
    return torch.randint(-high, high + 1, shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def slice_case(device, op, start, col):
    fn = ttnn.experimental.quasar.padded_slice
    if op.endswith("partial"):
        # The second row of 32-row shards holds only the slice's last 16 rows; the output is the 48-row slice.
        x = values((1, 1, 48, 96), op, start)
        mem = memory("BLOCK_SHARDED", start, (2, 2), (32, 32))
        y = fn(put(device, x, rm=True), [0, 0, 0, 0], [1, 1, 48, 64], memory_config=mem)
        return [compare(y, x[..., :48, :64])]
    x = values((1, 1, 96, 96), op, start, salt=int(col))
    mem = memory("BLOCK_SHARDED", start, (2, 2), (32, 32), col=col)
    y = fn(put(device, x, rm=True), [0, 0, 0, 0], [1, 1, 64, 64], memory_config=mem)
    return [compare(y, x[..., :64, :64])]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [(op, origin, col) for op in ["quasar_slice_rm"] for origin in FOUR_ORIGINS for col in [False, True]] + [
    (op, origin, False) for op in ["quasar_slice_rm_partial"] for origin in [(0, 0), (2, 3)]
]


@pytest.mark.parametrize("op,origin,col", CASES)
def test_quasar_padded_slice_shard_grid_origin(device, op, origin, col):
    grid = device.compute_with_storage_grid_size()
    if origin[0] + 2 > grid.x or origin[1] + 2 > grid.y:
        pytest.skip(f"needs a {origin[0] + 2}x{origin[1] + 2} worker grid; the device has {grid.x}x{grid.y}")
    outputs = slice_case(device, op, origin, col)
    assert all(output["wrong"] == 0 for output in outputs), outputs
