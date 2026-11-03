"""ttnn.group_norm on a HEIGHT_SHARDED input whose 2x2 shard grid is one rectangle written as two ranges.

groupnorm_sharded_program_factory.cpp merges the shard grid (`grid.merge_ranges()`), walks the merged
rectangle with corerange_to_cores(all_cores, n, orientation == ROW_MAJOR), and treats walk index k as shard k:
batch b is reduced over walk cores [b * cores_per_batch, (b + 1) * cores_per_batch). HEIGHT_SHARDED buffers place
shard k on corerange_to_cores(grid, n, row_major)[k] over the ranges as given (ROUND_ROBIN_1D), so ranges that run
against the orientation put rows of both batches in one reduction group. Validation (groupnorm_device_operation.cpp) only rejects grids with holes. One Blackhole chip or ttsim.

Input [2, 1, 64, 64] bf16 TILE (N=2, H*W=64, C=64), 2 groups, shard [32, 64]: each batch spans two cores. Batch 0
holds multiples of 1/8 in [-2, 2], batch 1 is 3 + 2x such values, so mixed statistics are far off. Expected: torch
group_norm (eps 1e-5) of ttnn.to_torch(input). The result is not exact (bf16 output, rsqrt), so a value is wrong if
|got - want| > 0.05 + 0.03 * |want| (or NaN); HiFi4, fp32 dest. Both statistics backends are run."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR


def grid(*ranges):  # a list keeps the ranges in the order given
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def error_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None)
    if line is None:
        line = lines[lines.index("info:") + 1] if "info:" in lines[:-1] else lines[0]
    return re.sub(r"\S*/", "", line)[:170]


CASES = [  # (name, shard grid, orientation)
    ("2x2 at (0, 0), ROW_MAJOR, control", grid(((0, 0), (1, 1))), RM),
    ("2x2 at (0, 0), COL_MAJOR, control", grid(((0, 0), (1, 1))), CM),
    ("2x2 at (1, 1), ROW_MAJOR (offset, audit says handled)", grid(((1, 1), (2, 2))), RM),
    ("2x2 as two one-column ranges, ROW_MAJOR", grid(((0, 0), (0, 1)), ((1, 0), (1, 1))), RM),
    ("2x2 as two one-row ranges, COL_MAJOR", grid(((0, 0), (1, 0)), ((0, 1), (1, 1))), CM),
]

device = ttnn.open_device(device_id=0)
ck = ttnn.init_device_compute_kernel_config(
    device.arch(),
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)
torch.manual_seed(0)
host = torch.randint(-16, 17, (2, 1, 64, 64)) / 8
host[1] = 3 + 2 * host[1]
host = host.to(torch.bfloat16)
for name, shard_grid, orientation in CASES:
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(shard_grid, [32, 64], orientation)
    )
    for welford in (False, True):
        op = f"group_norm(use_welford={welford})"
        x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        xt = ttnn.to_torch(x).float()
        want = torch.nn.functional.group_norm(xt.permute(0, 3, 1, 2), 2, eps=1e-5).permute(0, 2, 3, 1)
        try:
            y = ttnn.group_norm(
                x,
                num_groups=2,
                epsilon=1e-5,
                memory_config=mem,
                compute_kernel_config=ck,
                inplace=False,
                use_welford=welford,
            )
            got = ttnn.to_torch(y).float().reshape(want.shape)
            wrong = int((~((got - want).abs() <= 0.05 + 0.03 * want.abs())).sum())
            print(f"{op}, {name}: {wrong} of {want.numel()} values wrong", flush=True)
        except RuntimeError as e:
            print(f"{op}, {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
