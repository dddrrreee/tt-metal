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


def case(device, op, start, col, expect_error):
    def vals(shape):
        return values(shape, op, start, high=48)

    reshard = ttnn.experimental.quasar.reshard
    if op.endswith("reshard"):
        x = vals((1, 1, 8, 96))
        a = put(device, x, memory("WIDTH_SHARDED", start, (2, 1), (8, 64)), rm=True)
        y = reshard(a, memory("WIDTH_SHARDED", start, (3, 1), (8, 32)))
        return [compare(y, x)]
    if op.endswith("reshard_spare"):
        # Two shards on a three-core grid whose last core holds nothing.
        x = vals((1, 1, 8, 96))
        grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(start[0] + 1, start[1])),
                ttnn.CoreRange(ttnn.CoreCoord(start[0] + 3, start[1]), ttnn.CoreCoord(start[0] + 3, start[1])),
            }
        )
        spec = ttnn.ShardSpec(grid, (8, 64), ttnn.ShardOrientation.ROW_MAJOR)
        a = put(device, x, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, spec), rm=True)
        y = reshard(a, memory("WIDTH_SHARDED", start, (3, 1), (8, 32)))
        return [compare(y, x)]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [(op, origin, False) for op in ["quasar_reshard", "quasar_reshard_spare"] for origin in FOUR_ORIGINS]


@pytest.mark.parametrize("op,origin,col", CASES)
def test_quasar_reshard_shard_grid_origin(device, op, origin, col, expect_error):
    outputs = case(device, op, origin, col, expect_error)
    assert all(output["wrong"] == 0 for output in outputs), outputs
