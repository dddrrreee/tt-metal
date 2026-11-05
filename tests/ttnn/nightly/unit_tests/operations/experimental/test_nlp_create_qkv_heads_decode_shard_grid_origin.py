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

    if op in ["decode_tiles_32x16", "decode_tiles_transposed", "decode_tiles_16x32_wide_shard"]:
        # Narrower or transposed tiles mix heads, and the sharded reader steps a 32x32 tile within a shard: 16x32 tiles
        # in a shard four tiles wide are rejected too.
        x = vals((1, 1, 2, 128))
        tile = {
            "decode_tiles_32x16": ttnn.Tile((32, 16)),
            "decode_tiles_transposed": ttnn.Tile((32, 32), transpose_tile=True),
            "decode_tiles_16x32_wide_shard": ttnn.Tile((16, 32)),
        }[op]
        a = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory("WIDTH_SHARDED", (0, 0), (1, 1), (16 if op.endswith("wide_shard") else 32, 128)),
            tile=tile,
        )
        with expect_error(RuntimeError, "nlp_create_qkv_heads_decode needs untransposed tiles 32 wide"):
            ttnn.experimental.nlp_create_qkv_heads_decode(
                a, num_heads=2, num_kv_heads=1, memory_config=memory("HEIGHT_SHARDED", start, (2, 1), (32, 32))
            )
        return []
    if op in ["decode_tiles_16x32_sharded_one", "decode_tiles_16x32_sharded_tall"]:
        # 16x32 tiles in shards one tile wide: one tile per core for two users, or two stacked tiles for 32 users,
        # laid out as one 32x32 tile. The reader never steps within a shard.
        users = 32 if op.endswith("tall") else 2
        x = vals((1, 1, users, 128))
        a = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory("WIDTH_SHARDED", (0, 0), (4, 1), (16 if users == 2 else 32, 32)),
            tile=ttnn.Tile((16, 32)),
        )
        om = memory("HEIGHT_SHARDED", start, (8, 4) if users == 32 else (2, 1), (32, 32))
        ys = ttnn.experimental.nlp_create_qkv_heads_decode(a, num_heads=2, num_kv_heads=1, memory_config=om)
        for y in ys:
            assert str(y.memory_config().shard_spec.grid) == str(om.shard_spec.grid), (y.memory_config(), om)
        refs = [
            x[..., :64].reshape(1, users, 2, 32),
            x[..., 64:96].reshape(1, users, 1, 32),
            x[..., 96:].reshape(1, users, 1, 32),
        ]
        return [compare(y, r) for y, r in zip(ys, refs)]
    if op in ["decode_interleaved_out", "decode_sharded_out", "decode_tiles_16x32"]:
        # decode_tiles_16x32: an interleaved input in 16x32 tiles; the reader copies whole 32-wide tiles, as with 32x32
        # ones.
        x = vals((1, 1, 2, 128))
        if op == "decode_tiles_16x32":
            a = ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, tile=ttnn.Tile((16, 32))
            )
        else:
            im = memory("WIDTH_SHARDED", (0, 0), (1, 1), (32, 128)) if "sharded" in op else ttnn.DRAM_MEMORY_CONFIG
            a = put(device, x, im)
        om = memory("HEIGHT_SHARDED", start, (2, 1), (32, 32))
        ys = ttnn.experimental.nlp_create_qkv_heads_decode(a, num_heads=2, num_kv_heads=1, memory_config=om)
        # Q, K and V are on the requested grid, one user per core.
        for y in ys:
            assert str(y.memory_config().shard_spec.grid) == str(om.shard_spec.grid), (y.memory_config(), om)
        refs = [x[..., :64].reshape(1, 2, 2, 32), x[..., 64:96].reshape(1, 2, 1, 32), x[..., 96:].reshape(1, 2, 1, 32)]
        return [compare(y, r) for y, r in zip(ys, refs)]


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = (
    [(op, origin, False) for op in ["decode_interleaved_out", "decode_sharded_out"] for origin in FOUR_ORIGINS]
    + [
        (op, origin, False)
        for op in [
            "decode_tiles_32x16",
            "decode_tiles_transposed",
            "decode_tiles_16x32_wide_shard",
            "decode_tiles_16x32",
            "decode_tiles_16x32_sharded_one",
        ]
        for origin in [(0, 0), (2, 3)]
    ]
    + [
        # 32 users on an 8x4 output grid: origins that keep it on an 8x8 Wormhole grid.
        (op, origin, False)
        for op in ["decode_tiles_16x32_sharded_tall"]
        for origin in [(0, 0), (0, 1)]
    ]
)


@pytest.mark.parametrize("op,origin,col", CASES)
def test_nlp_create_qkv_heads_decode_shard_grid_origin(device, op, origin, col, expect_error):
    outputs = case(device, op, origin, col, expect_error)
    assert all(output["wrong"] == 0 for output in outputs), outputs
