"""ttnn.experimental.fast_reduce_nc(x, dims=[0]) with ND-sharded input and output and more output shards than cores.

With dim 0, ROW_MAJOR ROUND_ROBIN_1D ND shards on both tensors and more output shards than device cores,
fast_reduce_nc_program_factory.cpp takes the core groups and per-core shard counts from the output buffer
(dspec.core_groups_tuple(), which follows corerange_to_cores(grid, n, ROW_MAJOR)), but the factory names core i
{i % gx, i / gx} over the device compute grid and gives it shards i, i + n, ..., throwing "Core not in
specified core ranges." when that core is in neither group. The kernels address tiles by page id, so a walk that
stays inside the grid but in another order hands the extra shard of group 1 to the wrong cores: some output tiles
are never written and others past the end are. Validation only checks that the shard grids lie inside the device grid
(reduce_op_validation.cpp). One Blackhole chip or ttsim.

Input [2, 1, 32 * S, 32] bf16 TILE with S = gx * gy + 10 output tiles; input and output ND-sharded [1, 1, 32, 32] on
the same grid. Values are multiples of 1/8 in [-2, 2] with a new seed per case, so the sum over dim 0 is exact. An L1
guard tensor is allocated first so that reads past the end of a shard region stay in allocated L1."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM = ttnn.ShardOrientation.ROW_MAJOR


def grid(*ranges):  # a list keeps the ranges in the order given
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def error_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None)
    if line is None:
        line = lines[lines.index("info:") + 1] if "info:" in lines[:-1] else lines[0]
    return re.sub(r"\S*/", "", line)[:170]


device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
gx, gy = g.x, g.y
S = gx * gy + 10
CASES = [  # (name, shard grid of input and output)
    (f"row (0, 0)-({gx - 1}, 0), control", grid(((0, 0), (gx - 1, 0)))),
    (f"rows (0, 0)-({gx - 1}, 1), control", grid(((0, 0), (gx - 1, 1)))),
    ("cores (1, 0)-(4, 0)", grid(((1, 0), (4, 0)))),
    (f"row (0, 1)-({gx - 1}, 1)", grid(((0, 1), (gx - 1, 1)))),
    (f"rows 0 and 1 of width {gx} as two ranges, row 1 first", grid(((0, 1), (gx - 1, 1)), ((0, 0), (gx - 1, 0)))),
]
guard = ttnn.from_torch(
    torch.zeros(1, 1, 32, 32 * 4 * gx * gy, dtype=torch.bfloat16),
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
for seed, (name, shard_grid) in enumerate(CASES):
    torch.manual_seed(seed)
    mem = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 32, 32]), shard_grid, RM))
    host = (torch.randint(-16, 17, (2, 1, 32 * S, 32)) / 8).to(torch.bfloat16)
    x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    xt = ttnn.to_torch(x).float()
    want = xt.sum(dim=0, keepdim=True)
    try:
        y = ttnn.experimental.fast_reduce_nc(x, dims=[0], memory_config=mem)
        wrong = int((ttnn.to_torch(y).float().reshape(want.shape) != want).sum())
        print(f"fast_reduce_nc dims=[0], {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"fast_reduce_nc dims=[0], {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
