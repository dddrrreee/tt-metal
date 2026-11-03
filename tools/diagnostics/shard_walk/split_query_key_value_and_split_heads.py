"""ttnn.experimental.split_query_key_value_and_split_heads (BERT sharded path) on a BLOCK-sharded grid off (0, 0).

Variant of the walk: no index walk. The sharded factory
(split_query_key_value_and_split_heads_sharded_program_factory.cpp) takes num_h_cores = bbox.end.y + 1 and
num_w_cores = bbox.end.x + 1 for ROW_MAJOR (end.x + 1 / end.y + 1 for COL_MAJOR) and sizes each core's block as
block_h = B*S / num_h_cores, block_w = W / num_w_cores; every DFB borrows the core's own shard and the kernels read
only their own core. Buffers use the real grid: input shard [B*S / real rows, W / real cols]. Validation (device op) requires one range, bbox end inside the grid and BLOCK_SHARDED, but never start == (0, 0); the debug-only TT_ASSERTs would catch it. Per core the block holds 2 heads: [q_2c, q_2c+1, k_2c, k_2c+1, v_2c, v_2c+1]
(reader/writer kernels; tensor stride = block_w / 3). Outputs: Q, V [B, H, S, hd], K [B, H, hd, S]. Values are
multiples of 1/8; the op only moves data (K through a tile transpose): exact comparison."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
L1 = ttnn.BufferType.L1
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
B, S, NH, HD = 2, 384, 8, 64  # BERT-style block: 384 x 384 per core, 2 heads per core
W = 3 * NH * HD


def grid(*ranges):  # a list keeps the range order as written
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def eighths(shape):
    return (torch.randint(-16, 17, shape) / 8).to(torch.bfloat16)


def err_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None)
    if line is None:
        i = next((j for j, l in enumerate(lines) if l == "info:"), None)
        line = lines[i + 1] if i is not None and i + 1 < len(lines) else lines[0]
    return re.sub(r"/\S*/", "", line)[:200]


# (name, input grid, orientation); every grid is 2 batch rows x 4 width blocks of [384, 384]
CASES = [
    ("ROW_MAJOR on (0, 0)-(3, 1), control", ((0, 0), (3, 1)), RM),
    ("COL_MAJOR on (0, 0)-(1, 3), control", ((0, 0), (1, 3)), CM),
    ("ROW_MAJOR on (0, 1)-(3, 2)", ((0, 1), (3, 2)), RM),
    ("COL_MAJOR on (1, 0)-(2, 3)", ((1, 0), (2, 3)), CM),
]

device = ttnn.open_device(device_id=0)
dev_grid = device.compute_with_storage_grid_size()
for case, (name, rng, orientation) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    mem = ttnn.MemoryConfig(BLOCK, L1, ttnn.ShardSpec(grid(rng), [S, W // 4], orientation))
    x = ttnn.from_torch(eighths((B, 1, S, W)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    xt = ttnn.to_torch(x).float()[:, 0]  # [B, S, W]
    blk = W // 4  # one core's columns: two heads each of q, k, v
    sec = lambda h, t: xt[:, :, (h // 2) * blk + (2 * t + h % 2) * HD:(h // 2) * blk + (2 * t + h % 2 + 1) * HD]
    want = [torch.stack([sec(h, 0) for h in range(NH)], 1),
            torch.stack([sec(h, 1).transpose(-2, -1) for h in range(NH)], 1),
            torch.stack([sec(h, 2) for h in range(NH)], 1)]
    try:
        got = ttnn.experimental.split_query_key_value_and_split_heads(
            x, CC(dev_grid.x, dev_grid.y), memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG, num_heads=NH)
        wrong = sum(int((ttnn.to_torch(g).float() != w).sum()) for g, w in zip(got, want))
        total = sum(w.numel() for w in want)
        print(f"split_query_key_value_and_split_heads, {name}: {wrong} of {total} values wrong", flush=True)
    except RuntimeError as e:
        print(f"split_query_key_value_and_split_heads, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
