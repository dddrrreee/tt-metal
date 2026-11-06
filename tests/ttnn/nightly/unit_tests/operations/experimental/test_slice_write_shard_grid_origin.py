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


def slice_write_case(device, op, start, col, expect_error):
    rm = "rm" in op
    if op.endswith("partial"):
        # A 48-wide source in two 32-wide shards writes only the slice's columns of a 96-wide destination.
        x = values((1, 1, 32, 48), op, start)
        a = put(device, x, memory("BLOCK_SHARDED", start, (2, 1), (32, 32)), rm=True)
        out = put(device, torch.full((1, 1, 32, 96), 7, dtype=torch.bfloat16), rm=True)
        fn = ttnn.experimental.slice_write
        fn(a, out, [0, 0, 0, 0], [1, 1, 32, 48], [1, 1, 1, 1])
        return [compare(out, torch.cat([x, torch.full((1, 1, 32, 48), 7, dtype=torch.bfloat16)], dim=-1))]
    if op.endswith("spare_rows"):
        # A 32-row source on three rows of 32-row shards: the two spare rows of cores write nothing.
        x = values((1, 1, 32, 64), op, start)
        a = put(device, x, memory("BLOCK_SHARDED", start, (2, 3), (32, 32)), rm=rm)
        out = put(device, torch.full((1, 1, 32, 64), 7, dtype=torch.bfloat16), rm=rm)
        fn = ttnn.experimental.slice_write
        fn(a, out, [0, 0, 0, 0], [1, 1, 32, 64], [1, 1, 1, 1])
        return [compare(out, x)]
    if op.endswith("empty"):
        # An empty slice, rows 16 to 16, returns the destination unchanged although it ends inside a tile.
        a = put(device, values((1, 1, 32, 32), op, start), memory("HEIGHT_SHARDED", start, (1, 1), (32, 32)))
        dest = torch.full((1, 1, 64, 32), 7, dtype=torch.bfloat16)
        out = put(device, dest)
        ttnn.experimental.slice_write(a, out, [0, 0, 16, 0], [1, 1, 16, 32], [1, 1, 1, 1])
        return [compare(out, dest)]
    if op.endswith(("partial_end", "end_at_dest")):
        # A 48-row tiled source: half of its second tile row is padding, which lands on rows 48-63 of a taller
        # destination, or in the padding of one that ends at row 48.
        x = values((1, 1, 48, 32), op, start)
        a = put(device, x, memory("HEIGHT_SHARDED", start, (2, 1), (32, 32)))
        out = put(device, torch.full((1, 1, 96 if op.endswith("partial_end") else 48, 32), 7, dtype=torch.bfloat16))
        fn = ttnn.experimental.slice_write
        if op.endswith("partial_end"):
            with expect_error(RuntimeError, "must end on a tile boundary or at the destination's end"):
                fn(a, out, [0, 0, 0, 0], [1, 1, 48, 32], [1, 1, 1, 1])
            return []
        fn(a, out, [0, 0, 0, 0], [1, 1, 48, 32], [1, 1, 1, 1])
        return [compare(out, x)]
    if op.endswith("narrow"):
        # A 16-wide shard of 32x16 tiles: the tiled path counts 32-wide tiles, so it rejects the shard.
        x = values((1, 1, 32, 16), op, start)
        mem = memory("HEIGHT_SHARDED", start, (1, 1), (32, 16))
        a = ttnn.from_torch(
            x, device=device, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile((32, 16)), dtype=ttnn.bfloat16, memory_config=mem
        )
        out = put(device, torch.zeros((1, 1, 32, 16), dtype=torch.bfloat16))
        with expect_error(RuntimeError, "a whole number of 32-column tiles wide"):
            ttnn.experimental.slice_write(a, out, [0, 0, 0, 0], [1, 1, 32, 16], [1, 1, 1, 1])
        return []
    if op.endswith(("spare_cols", "wide_shard", "wide_rows")):
        # TILE sources into a 96-wide destination: spare_cols is 64 wide on three columns of 32-wide shards, the third
        # past the slice; wide_shard is 32 wide in one 64-wide shard, half past the slice; wide_rows has two tile rows.
        rows = 64 if op.endswith("wide_rows") else 32
        width, size, shard = (64, (3, 1), (32, 32)) if op.endswith("spare_cols") else (32, (1, 1), (rows, 64))
        x = values((1, 1, rows, width), op, start)
        a = put(device, x, memory("BLOCK_SHARDED", start, size, shard))
        out = put(device, torch.full((1, 1, rows, 96), 7, dtype=torch.bfloat16))
        fn = ttnn.experimental.slice_write
        fn(a, out, [0, 0, 0, 0], [1, 1, rows, width], [1, 1, 1, 1])
        return [compare(out, torch.cat([x, torch.full((1, 1, rows, 96 - width), 7, dtype=torch.bfloat16)], dim=-1))]
    x = values((1, 1, 64, 64), op, start, salt=int(col))
    a = put(device, x, memory("BLOCK_SHARDED", start, (2, 2), (32, 32), col=col), rm=rm)
    out = put(device, torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16), rm=rm)
    fn = ttnn.experimental.slice_write
    fn(a, out, [0, 0, 0, 0], [1, 1, 64, 64], [1, 1, 1, 1])
    return [compare(out, x)]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [
    (op, origin, col)
    for op in ["slice_write_rm", "slice_write_tile"]
    for origin in FOUR_ORIGINS
    for col in [False, True]
] + [
    (op, origin, False)
    for op in [
        "slice_write_partial",
        "slice_write_rm_spare_rows",
        "slice_write_tile_spare_rows",
        "slice_write_tile_spare_cols",
        "slice_write_tile_wide_shard",
        "slice_write_tile_wide_rows",
        "slice_write_tile_partial_end",
        "slice_write_tile_end_at_dest",
        "slice_write_tile_empty",
        "slice_write_tile_narrow",
    ]
    for origin in [(0, 0), (2, 3)]
]


@pytest.mark.parametrize("op,origin,col", CASES)
def test_slice_write_shard_grid_origin(device, op, origin, col, expect_error):
    outputs = slice_write_case(device, op, origin, col, expect_error)
    assert all(output["wrong"] == 0 for output in outputs), outputs
