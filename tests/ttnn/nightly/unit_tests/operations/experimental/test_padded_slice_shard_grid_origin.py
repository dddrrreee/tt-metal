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


def slice_case(device, op, start, col, expect_error):
    rm = "rm" in op
    if op.endswith(("crop", "spare_cols")):
        # crop: the top-left 64x64 of a 96x96 tiled input, its width cut too; spare_cols: all of a 64-wide tiled input
        # on four columns of 32-wide shards, two of them past it.
        size = (96, 96) if op.endswith("crop") else (64, 64)
        x = values((1, 1, *size), op, start)
        mem = memory("BLOCK_SHARDED", start, (2, 2) if op.endswith("crop") else (4, 2), (32, 32))
        y = ttnn.experimental.padded_slice(put(device, x), [0, 0, 0, 0], [1, 1, 64, 64], memory_config=mem)
        return [compare(y, x[..., :64, :64])]
    if op.endswith(("crop_offset", "crop_unaligned")):
        # Columns 64-95 of a 96-wide input, from a tile column; columns 16-47 of a 64-wide one, from inside a tile.
        aligned = op.endswith("offset")
        width, begin = (96, 64) if aligned else (64, 16)
        x = values((1, 1, 32, width), op, start)
        mem = memory("BLOCK_SHARDED", start, (1, 1), (32, 64 if aligned else 32))
        args = (put(device, x), [0, 0, 0, begin], [1, 1, 32, begin + 32])
        if not aligned:
            with expect_error(RuntimeError, "must start at a multiple of 32 columns"):
                ttnn.experimental.padded_slice(*args, memory_config=mem)
            return []
        y = ttnn.experimental.padded_slice(*args, memory_config=mem)
        return [compare(y, x[..., begin : begin + 32])]
    if op.endswith("crop_rows"):
        # Rows 16-31 of a 256-wide tiled input, from inside a tile row, in one 16-row shard that ends with the slice.
        x = values((1, 1, 32, 256), op, start)
        mem = memory("BLOCK_SHARDED", start, (1, 1), (16, 256))
        y = ttnn.experimental.padded_slice(put(device, x), [0, 0, 16, 0], [1, 1, 32, 256], memory_config=mem)
        return [compare(y, x[..., 16:32, :])]
    if op.endswith(("crop_batches", "wide_shard")):
        # crop_batches: two batches cut in height and width into 64-wide shards; wide_shard: half of a 64-wide input
        # in one 64-wide shard.
        batches = op.endswith("batches")
        shape, end = ((2, 1, 64, 128), [2, 1, 32, 96]) if batches else ((1, 1, 32, 64), [1, 1, 32, 32])
        x = values(shape, op, start)
        mem = memory("BLOCK_SHARDED", start, (2, 1) if batches else (1, 1), (64, 64) if batches else (32, 64))
        y = ttnn.experimental.padded_slice(put(device, x), [0, 0, 0, 0], end, memory_config=mem)
        # The op returns its rows flattened over the batches.
        return [compare(y, x[: end[0], : end[1], : end[2], : end[3]].reshape(1, 1, -1, end[3]))]
    if op.endswith(("height_crop", "height_wide")):
        # A HEIGHT_SHARDED output is as wide as its shards: columns 0-31 of a 96-wide tiled input into 32-wide shards,
        # and into 64-wide ones, whose columns past the slice the reader would fill from the input (rejected).
        wide = op.endswith("wide")
        x = values((1, 1, 32, 96), op, start)
        mem = memory("HEIGHT_SHARDED", start, (1, 1), (32, 64 if wide else 32))
        args = (put(device, x), [0, 0, 0, 0], [1, 1, 32, 32])
        if wide:
            with expect_error(RuntimeError, "would fill the"):
                ttnn.experimental.padded_slice(*args, memory_config=mem)
            return []
        y = ttnn.experimental.padded_slice(*args, memory_config=mem)
        return [compare(y, x[..., :32])]
    if op.endswith(("partial", "spare_rows")):
        # partial: the second row of 32-row shards holds only the slice's last 16 rows; spare_rows: a 32-row slice, so
        # the second row of shards holds none of it. The output is the slice.
        rows = 32 if op.endswith("spare_rows") else 48
        x = values((1, 1, 48, 96), op, start)
        fn = ttnn.experimental.padded_slice
        mem = memory("BLOCK_SHARDED", start, (2, 2), (32, 32))
        y = fn(put(device, x, rm=True), [0, 0, 0, 0], [1, 1, rows, 64], memory_config=mem)
        return [compare(y, x[..., :rows, :64])]
    x = values((1, 1, 96, 96 if rm else 64), op, start, salt=int(col))
    mem = memory("BLOCK_SHARDED", start, (2, 2), (32, 32), col=col)
    fn = ttnn.experimental.padded_slice
    y = fn(put(device, x, rm=rm), [0, 0, 0, 0], [1, 1, 64, 64], memory_config=mem)
    return [compare(y, x[..., :64, :64])]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [(op, origin, col) for op in ["slice_rm", "slice_tile"] for origin in FOUR_ORIGINS for col in [False, True]] + [
    (op, origin, False)
    for op in [
        "slice_rm_partial",
        "slice_rm_spare_rows",
        "slice_tile_crop",
        "slice_tile_crop_batches",
        "slice_tile_crop_offset",
        "slice_tile_crop_unaligned",
        "slice_tile_crop_rows",
        "slice_tile_wide_shard",
        "slice_tile_spare_cols",
        "slice_tile_height_crop",
        "slice_tile_height_wide",
    ]
    for origin in [(0, 0), (2, 3)]
]


@pytest.mark.parametrize("op,origin,col", CASES)
def test_padded_slice_shard_grid_origin(device, op, origin, col, expect_error):
    # The widest case, spare_cols, needs four columns and two rows of cores from its origin.
    grid = device.compute_with_storage_grid_size()
    if origin[0] + 4 > grid.x or origin[1] + 2 > grid.y:
        pytest.skip(f"needs a {origin[0] + 4}x{origin[1] + 2} worker grid; the device has {grid.x}x{grid.y}")
    outputs = slice_case(device, op, origin, col, expect_error)
    assert all(output["wrong"] == 0 for output in outputs), outputs
