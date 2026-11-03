"""ttnn.experimental.create_qkv_heads with a BLOCK-sharded fused QKV input whose grid does not start at (0, 0).

Variant of the walk: no index walk, but the per-core block sizes count cores from (0, 0). The factory
(create_qkv_heads_program_factory.cpp) takes num_h_cores = bbox.end.y + 1 and num_w_cores = bbox.end.x + 1
(ROW_MAJOR) and derives groups_per_block = groups / num_w_cores, block_wt = Kt / num_w_cores and
block_ht = Mt / num_h_cores from them; every DFB borrows the core's own shard and the kernel reads only its own
core. The buffers use the real grid: the input shard is [B*S / rows, W / cols] and the HEIGHT-sharded outputs
are spread over bbox.size() cores (device op). Validation (device op) requires batch == end.y + 1
and W % ((end.x + 1) * 32) == 0 but never start == (0, 0). Layout as in the op's test: per KV group,
[q heads of the group, k head, v head]. Values are multiples of 1/8 and the op only moves data: exact comparison."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM = ttnn.ShardOrientation.ROW_MAJOR
L1 = ttnn.BufferType.L1
BLOCK, HEIGHT = ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED


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


# (name, batch, seq, num_q_heads, num_kv_heads, head_dim, input grid as ((x0, y0), (x1, y1)))
CASES = [
    ("batch 3, 2 q / 2 kv heads of 64, on (0, 0)-(1, 2), control", 3, 64, 2, 2, 64, ((0, 0), (1, 2))),
    ("batch 3, 6 q / 6 kv heads of 32, on (0, 0)-(1, 2), control", 3, 64, 6, 6, 32, ((0, 0), (1, 2))),
    ("batch 3, 2 q / 2 kv heads of 64, on (0, 1)-(1, 2)", 3, 64, 2, 2, 64, ((0, 1), (1, 2))),
    ("batch 3, 6 q / 6 kv heads of 32, on (1, 0)-(2, 2)", 3, 64, 6, 6, 32, ((1, 0), (2, 2))),
]

device = ttnn.open_device(device_id=0)
for case, (name, b, s, nq, nkv, hd, ((x0, y0), (x1, y1))) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    qpg = nq // nkv
    per = (qpg + 2) * hd  # one KV group: [qpg q heads, k, v]
    w = per * nkv
    rows, cols = y1 - y0 + 1, x1 - x0 + 1
    mem = ttnn.MemoryConfig(BLOCK, L1, ttnn.ShardSpec(grid(((x0, y0), (x1, y1))), [b * s // rows, w // cols], RM))
    x = ttnn.from_torch(eighths((b, 1, s, w)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    xt = ttnn.to_torch(x).float()[:, 0]  # [b, s, w]
    col = lambda g, slot: xt[:, :, g * per + slot * hd:g * per + (slot + 1) * hd]
    want = [torch.stack([col(h // qpg, h % qpg) for h in range(nq)], 1),
            torch.stack([col(g, qpg) for g in range(nkv)], 1),
            torch.stack([col(g, qpg + 1) for g in range(nkv)], 1)]
    try:
        got = ttnn.experimental.create_qkv_heads(x, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False,
                                                 memory_config=ttnn.MemoryConfig(HEIGHT, L1))
        wrong = sum(int((ttnn.to_torch(g).float() != w_).sum()) for g, w_ in zip(got, want))
        total = sum(w_.numel() for w_ in want)
        print(f"create_qkv_heads, {name}: {wrong} of {total} values wrong", flush=True)
    except RuntimeError as e:
        print(f"create_qkv_heads, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
