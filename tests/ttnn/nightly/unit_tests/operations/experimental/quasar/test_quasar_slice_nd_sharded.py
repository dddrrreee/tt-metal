# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


def nd_config(shard_shape, last_core=(7, 1), strategy=ttnn.ShardDistributionStrategy.CONTIGUOUS_1D):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*last_core))})
    spec = ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, ttnn.ShardOrientation.ROW_MAJOR, strategy)
    return ttnn.MemoryConfig(ttnn.BufferType.L1, spec)


def wrong(actual, expected):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    return int((actual != expected).sum())


# A row step of 2 takes the stride factory, which reads whole rows.
@pytest.mark.parametrize("row_step", [1, 2])
# Shard rows of 16 and 32 B, and of 8 B, below the L1 alignment; shards one row high squeeze to a rank-one distribution.
@pytest.mark.parametrize("shard", [[1, 1, 8, 8], [1, 1, 8, 16], [1, 1, 8, 4], [1, 1, 1, 8]])
@pytest.mark.parametrize("output", ["dram", "derived"])
@pytest.mark.parametrize("rows, cols", [((0, 32), (0, 64)), ((12, 44), (0, 64)), ((0, 32), (16, 48))])
def test_quasar_slice_nd_sharded(device, shard, output, rows, cols, row_step):
    # A row-major ND-sharded tensor's pages are shard rows, so each of its rows spans several pages (#64473).
    x = ((torch.arange(64 * 64) % 29 + 1) / 8).reshape(1, 1, 64, 64).bfloat16()
    want = x[:, :, rows[0] : rows[1] : row_step, cols[0] : cols[1]]
    earlier = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=nd_config(shard))
    t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=nd_config(shard))
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG} if output == "dram" else {}
    begins, ends = [0, 0, rows[0], cols[0]], [1, 1, rows[1], cols[1]]
    y = ttnn.experimental.quasar.slice(t, begins, ends, [1, 1, row_step, 1], **kwargs)
    # The slice is exact, and neither the input nor an earlier L1 tensor changes.
    assert wrong(ttnn.to_torch(y), want) == 0
    assert wrong(ttnn.to_torch(t), x) == 0
    assert wrong(ttnn.to_torch(earlier), x) == 0


# A rank-one tensor is a single row: [128] in [8] or [16] shards on 8 cores is 16 or 8 shard pages.
@pytest.mark.parametrize("strategy", ["CONTIGUOUS_1D", "ROUND_ROBIN_1D"])
@pytest.mark.parametrize("shard", [[8], [16]])
@pytest.mark.parametrize("begin, end, step", [(0, 32, 1), (8, 40, 1), (0, 64, 2)])
def test_quasar_slice_nd_sharded_rank_one(device, shard, begin, end, step, strategy):
    x = ((torch.arange(128) % 29 + 1) / 8).bfloat16()
    want = x[begin:end:step]
    mc = nd_config(shard, (7, 0), getattr(ttnn.ShardDistributionStrategy, strategy))
    earlier = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    y = ttnn.experimental.quasar.slice(t, [begin], [end], [step], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert wrong(ttnn.to_torch(y), want) == 0
    assert wrong(ttnn.to_torch(t), x) == 0
    assert wrong(ttnn.to_torch(earlier), x) == 0


# An empty slice returns an empty tensor, as it does for other inputs.
@pytest.mark.parametrize(
    "shape, shard, last_core, begins, ends, output",
    [
        ([128], [8], (7, 0), [8], [8], "dram"),
        ([1, 1, 64, 64], [1, 1, 8, 8], (7, 1), [0, 0, 16, 16], [1, 1, 16, 48], "derived"),
    ],
)
def test_quasar_slice_nd_sharded_empty(device, shape, shard, last_core, begins, ends, output):
    x = ((torch.arange(64 * 64)[: torch.Size(shape).numel()] % 29 + 1) / 8).reshape(shape).bfloat16()
    t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=nd_config(shard, last_core))
    kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG} if output == "dram" else {}
    y = ttnn.experimental.quasar.slice(t, begins, ends, [1] * len(shape), **kwargs)
    assert list(y.shape) == [e - b for b, e in zip(begins, ends)]
    assert wrong(ttnn.to_torch(t), x) == 0


# Shards one row high under a shard dim above 1: the distribution squeezes the rows into its last dim.
def test_quasar_slice_nd_sharded_merged_rows(device):
    x = ((torch.arange(2 * 32 * 64) % 29 + 1) / 8).reshape(1, 2, 32, 64).bfloat16()
    t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=nd_config([1, 2, 1, 8]))
    y = ttnn.experimental.quasar.slice(
        t, [0, 0, 0, 0], [1, 2, 16, 64], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert wrong(ttnn.to_torch(y), x[:, :, 0:16, :]) == 0
    assert wrong(ttnn.to_torch(t), x) == 0
