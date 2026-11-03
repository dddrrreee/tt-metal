"""ttnn.experimental.create_qkv_heads_from_separate_tensors with BLOCK-sharded Q / KV inputs off the walked grid.

Variant of the walk: no index walk. The factory (create_qkv_heads_from_separate_tensors_program_factory.cpp)
takes num_h_cores = bbox.end.y + 1, num_w_cores = bbox.end.x + 1 (ROW_MAJOR) and derives
q_shard_ht = B*S / (num_h_cores * 32) and the heads per core from them; all DFBs borrow the core's own shards
(IN_Q, IN_KV, OUT_Q/K/V) on the Q grid, and the kernel reads only its own core. Buffers use the
real grid: input shards [B*S / rows, W / cols], outputs over bbox.size() cores (device op). Validation
(device op) requires batch == end.y + 1 but never start == (0, 0), and never compares the KV tensor's grid
or orientation with Q's, although IN_KV borrows the KV shard on each Q core. Layout as in the op's test: Q holds
heads side by side, KV holds [k_h, v_h] per head. Values are multiples of 1/8; the op only moves data: exact."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
L1 = ttnn.BufferType.L1
BLOCK, HEIGHT = ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
S, NH, HD = 64, 2, 128  # seq, heads (q and kv), head_dim: Q width 256, KV width 512


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


def block_mem(rng, b, w, orientation):  # the shard [rows, cols] follows the orientation
    (x0, y0), (x1, y1) = rng
    gx, gy = x1 - x0 + 1, y1 - y0 + 1
    hs, ws = (gy, gx) if orientation == RM else (gx, gy)
    return ttnn.MemoryConfig(BLOCK, L1, ttnn.ShardSpec(grid(rng), [b * S // hs, w // ws], orientation))


# (name, batch, Q grid, KV grid, KV orientation); Q is ROW_MAJOR
CASES = [
    ("batch 3, Q and KV on (0, 0)-(1, 2), control", 3, ((0, 0), (1, 2)), ((0, 0), (1, 2)), RM),
    ("batch 2, Q and KV on 2x2 at (0, 0), control", 2, ((0, 0), (1, 1)), ((0, 0), (1, 1)), RM),
    ("batch 3, Q and KV on (0, 1)-(1, 2)", 3, ((0, 1), (1, 2)), ((0, 1), (1, 2)), RM),
    ("batch 2, Q on 2x2 at (0, 0), KV COL_MAJOR there", 2, ((0, 0), (1, 1)), ((0, 0), (1, 1)), CM),
]

device = ttnn.open_device(device_id=0)
for case, (name, b, q_rng, kv_rng, kv_orient) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    q_in = ttnn.from_torch(eighths((b, 1, S, NH * HD)), layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=block_mem(q_rng, b, NH * HD, RM))
    kv_in = ttnn.from_torch(eighths((b, 1, S, 2 * NH * HD)), layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=block_mem(kv_rng, b, 2 * NH * HD, kv_orient))
    qt, kvt = ttnn.to_torch(q_in).float()[:, 0], ttnn.to_torch(kv_in).float()[:, 0]  # [b, S, width]
    want = [torch.stack([qt[:, :, h * HD:(h + 1) * HD] for h in range(NH)], 1),
            torch.stack([kvt[:, :, 2 * h * HD:(2 * h + 1) * HD] for h in range(NH)], 1),
            torch.stack([kvt[:, :, (2 * h + 1) * HD:(2 * h + 2) * HD] for h in range(NH)], 1)]
    try:
        got = ttnn.experimental.create_qkv_heads_from_separate_tensors(
            q_in, kv_in, num_heads=NH, num_kv_heads=NH, transpose_k_heads=False,
            memory_config=ttnn.MemoryConfig(HEIGHT, L1))
        wrong = sum(int((ttnn.to_torch(g).float() != w).sum()) for g, w in zip(got, want))
        total = sum(w.numel() for w in want)
        print(f"create_qkv_heads_from_separate_tensors, {name}: {wrong} of {total} values wrong", flush=True)
    except RuntimeError as e:
        print(f"create_qkv_heads_from_separate_tensors, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
