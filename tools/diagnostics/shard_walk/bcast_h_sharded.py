"""ttnn.bcast(a, b, ADD, BcastOpDim.H) with a WIDTH- or BLOCK-sharded `a`, on shard grids other than the walked one.

Both sharded bcast-H factories place core i's runtime args on a walk that starts at core (0, 0): WIDTH_SHARDED
walks rows as wide as the device's compute grid, BLOCK_SHARDED walks columns as tall as the first range.
Buffers place shard i on corerange_to_cores(grid, n, row_major)[i]. One Blackhole chip or ttsim. Values are
multiples of 1/8, so a + b is exact in bfloat16. b with one row per batch selects the non-optimised factory."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
WIDTH, BLOCK = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED


def grid(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges})


def eighths(shape):
    return (torch.randint(-16, 17, shape) / 8).to(torch.bfloat16)


ROW4 = grid(((0, 0), (3, 0)))
# (name, layout, shard grid, orientation, a shape, shard shape, b shape)
CASES = [
    ("WIDTH, cores (0, 0)-(3, 0), control", WIDTH, ROW4, RM, (1, 1, 32, 128), (32, 32), (1, 1, 1, 128)),
    ("WIDTH, cores (1, 0)-(4, 0)", WIDTH, grid(((1, 0), (4, 0))), RM, (1, 1, 32, 128), (32, 32), (1, 1, 1, 128)),
    ("WIDTH, 2x2 at (0, 0)", WIDTH, grid(((0, 0), (1, 1))), RM, (1, 1, 32, 128), (32, 32), (1, 1, 1, 128)),
    ("WIDTH, 2x2 as two one-row ranges, COL_MAJOR", WIDTH, grid(((0, 0), (1, 0)), ((0, 1), (1, 1))), CM,
     (1, 1, 32, 128), (32, 32), (1, 1, 1, 128)),
    ("WIDTH, cores (0, 0)-(3, 0), batch 2, control", WIDTH, ROW4, RM, (1, 2, 32, 128), (64, 32), (2, 1, 1, 128)),
    ("WIDTH, cores (1, 0)-(4, 0), batch 2", WIDTH, grid(((1, 0), (4, 0))), RM, (1, 2, 32, 128), (64, 32),
     (2, 1, 1, 128)),
    ("BLOCK, 2x2 at (0, 0), control", BLOCK, grid(((0, 0), (1, 1))), RM, (1, 1, 64, 64), (32, 32), (1, 1, 1, 64)),
    ("BLOCK, 2x2 at (0, 0), COL_MAJOR, control", BLOCK, grid(((0, 0), (1, 1))), CM, (1, 1, 64, 64), (32, 32),
     (1, 1, 1, 64)),
    ("BLOCK, 2x2 at (1, 1)", BLOCK, grid(((1, 1), (2, 2))), RM, (1, 1, 64, 64), (32, 32), (1, 1, 1, 64)),
]

device = ttnn.open_device(device_id=0)
gx = device.compute_with_storage_grid_size().x  # the WIDTH walk's row length
rows = grid(((0, 0), (gx - 1, 1)))
two_rows = grid(((0, 0), (gx - 1, 0)), ((0, 1), (gx - 1, 1)))
wide = (1, 1, 32, 64 * gx)
CASES += [
    (f"WIDTH, two full rows (0, 0)-({gx - 1}, 1), COL_MAJOR, control", WIDTH, rows, CM, wide, (32, 32),
     (1, 1, 1, 64 * gx)),
    (f"WIDTH, the same {2 * gx} cores as two one-row ranges, COL_MAJOR", WIDTH, two_rows, CM, wide, (32, 32),
     (1, 1, 1, 64 * gx)),
]
for name, layout, shard_grid, orientation, a_shape, shard_shape, b_shape in CASES:
    torch.manual_seed(0)
    a_host, b_host = eighths(a_shape), eighths(b_shape)
    want = a_host.float() + b_host.reshape(1, b_shape[0], 1, b_shape[-1]).float()
    mem = ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(shard_grid, list(shard_shape), orientation))
    a = ttnn.from_torch(a_host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    b = ttnn.from_torch(b_host, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        y = ttnn.bcast(a, b, ttnn.BcastOpMath.ADD, ttnn.BcastOpDim.H, memory_config=mem)
        wrong = int((ttnn.to_torch(y).float() != want).sum())
        print(f"bcast H, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        line = next((l for l in lines if l.startswith("Kernel")), lines[0])
        print(f"bcast H, {name}: {line[:170]}", flush=True)
ttnn.close_device(device)
