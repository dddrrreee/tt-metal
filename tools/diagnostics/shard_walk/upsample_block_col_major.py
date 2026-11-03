"""ttnn.upsample(x, 2) of a ROW_MAJOR-layout NHWC tensor that is BLOCK_SHARDED with COL_MAJOR orientation.

For BLOCK sharding the factory takes the channel split from the grid's y range and the NHW split from its x range,
and reads source sticks from CoreCoord(channel index, nhw core) (upsample_program_factory_multicore_sharded.cpp). That matches ROW_MAJOR block placement; COL_MAJOR places block shard (nhw k, channel c) on
(start.x + k, start.y + c). One Blackhole chip or ttsim. Values are multiples of 1/8; nearest upsample only copies."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR


def grid(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges})


CASES = [  # (name, grid, orientation, shard [NHW rows, channels]); input NHWC [1, 2, 4, 64], NHW = 8
    ("cores (0, 0)-(1, 0), ROW_MAJOR, control", grid(((0, 0), (1, 0))), RM, (8, 32)),
    ("cores (0, 0)-(1, 0), COL_MAJOR", grid(((0, 0), (1, 0))), CM, (4, 64)),
    ("2x2 at (1, 0), ROW_MAJOR, control", grid(((1, 0), (2, 1))), RM, (4, 32)),
    ("2x2 at (1, 0), COL_MAJOR", grid(((1, 0), (2, 1))), CM, (4, 32)),
]

device = ttnn.open_device(device_id=0, l1_small_size=24576)  # as test_upsample.py
for name, shard_grid, orientation, shard in CASES:
    torch.manual_seed(0)
    x = (torch.randint(-16, 17, (1, 2, 4, 64)) / 8).to(torch.bfloat16)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(shard_grid, list(shard), orientation))
    t = ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    want = x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    try:
        y = ttnn.to_torch(ttnn.upsample(t, 2)).reshape(want.shape)
        print(f"upsample(x, 2), {name}: {int((y != want).sum())} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        print(f"upsample(x, 2), {name}: {next((l for l in lines if l.startswith('Kernel')), lines[0])[:170]}",
              flush=True)
ttnn.close_device(device)
