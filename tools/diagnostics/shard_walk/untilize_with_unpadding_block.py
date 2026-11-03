"""ttnn.untilize_with_unpadding of a BLOCK_SHARDED TILE tensor to an interleaved output, on offset shard grids.

The sharded factory's BLOCK-to-interleaved writer takes each core's absolute x and y as its shard column and row
(untilize_with_unpadding_multi_core_sharded_program_factory.cpp), and skips cores past an end core that
is counted from (0, 0). Buffers place block shard (row r, column c) on (start.x + c, start.y + r) for ROW_MAJOR.
One Blackhole chip or ttsim. Values are multiples of 1/8, exact in bfloat16; untilize only moves values."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR


def grid(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges})


CASES = [  # (name, grid, orientation, tensor shape); one [32, 32] shard per core
    ("cores (0, 0)-(1, 0), control", grid(((0, 0), (1, 0))), RM, (1, 1, 32, 64)),
    ("cores (1, 0)-(2, 0)", grid(((1, 0), (2, 0))), RM, (1, 1, 32, 64)),
    ("cores (0, 1)-(1, 1)", grid(((0, 1), (1, 1))), RM, (1, 1, 32, 64)),
    ("2x2 at (0, 0), control", grid(((0, 0), (1, 1))), RM, (1, 1, 64, 64)),
    ("2x2 at (1, 1)", grid(((1, 1), (2, 2))), RM, (1, 1, 64, 64)),
    ("cores (0, 0)-(1, 0), COL_MAJOR, control", grid(((0, 0), (1, 0))), CM, (1, 1, 64, 32)),
    ("cores (1, 0)-(2, 0), COL_MAJOR", grid(((1, 0), (2, 0))), CM, (1, 1, 64, 32)),
]

device = ttnn.open_device(device_id=0)
for name, shard_grid, orientation, shape in CASES:
    torch.manual_seed(0)
    x = (torch.randint(-16, 17, shape) / 8).to(torch.bfloat16)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(shard_grid, [32, 32], orientation))
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    end = (0, 0, shape[2] - 2, shape[3] - 1)  # drop the last row
    want = x[:, :, : shape[2] - 1, :]
    try:
        y = ttnn.untilize_with_unpadding(t, end, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wrong = int((ttnn.to_torch(y) != want).sum())
        print(f"untilize_with_unpadding, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        print(f"untilize_with_unpadding, {name}: {next((l for l in lines if l.startswith('Kernel')), lines[0])[:170]}",
              flush=True)
ttnn.close_device(device)
