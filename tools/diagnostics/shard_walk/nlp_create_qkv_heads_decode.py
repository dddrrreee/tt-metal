"""ttnn.experimental.nlp_create_qkv_heads_decode with a memory_config whose shard grid is not the walked one.

The output Q/K/V grid is num_cores_to_corerangeset_in_subcoregrids(grid.ranges()[0].start, users, grid, true)
over the memory_config's shard grid (nlp_create_qkv_heads_decode.cpp, device op), so buffers put
user i on corerange_to_cores(q grid, n, true)[i]: range by range, from each range's real start. Both factories
give user i to `grid_to_cores(n, q_bbox.end.x + 1, q_bbox.end.y + 1, true)[i]`, a row-major walk from (0, 0)
over the bounding box (interleaved factory, sharded factory), and each core writes its user into
its own borrowed Q/K/V shards. Validation (device op) checks only HEIGHT_SHARDED and the core count. The
interleaved cases use a DRAM input; the sharded cases use the test's one-core WIDTH-sharded input at (0, 0), which
routing sends to the sharded factory (nlp_create_qkv_heads_decode.cpp). Values are multiples of 1/8 and the
op only moves data, so the comparison is exact."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM = ttnn.ShardOrientation.ROW_MAJOR
L1 = ttnn.BufferType.L1
WIDTH, HEIGHT = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
NQ, NKV, HD, USERS = 8, 1, 128, 32  # test_create_head_max_width_shard (8, 1, 128, 32)
W = (NQ + 2 * NKV) * HD


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


SHARDED_IN = ttnn.MemoryConfig(WIDTH, L1, ttnn.ShardSpec(grid(((0, 0), (0, 0))), [32, W], RM))
DEFAULT = ttnn.MemoryConfig(HEIGHT, L1)  # no shard spec: the whole device grid
GRIDS = [  # (name, output memory_config)
    ("default output grid, control", DEFAULT),
    ("output grid (0, 0)-(7, 3), control", ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(grid(((0, 0), (7, 3))), [32, HD], RM))),
    ("output grid (0, 1)-(7, 4)", ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(grid(((0, 1), (7, 4))), [32, HD], RM))),
    ("output grid (0, 0)-(3, 3) + (4, 0)-(7, 3)",
     ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(grid(((0, 0), (3, 3)), ((4, 0), (7, 3))), [32, HD], RM))),
]
CASES = [(f"interleaved input, {g}", ttnn.DRAM_MEMORY_CONFIG, m) for g, m in GRIDS]
CASES += [(f"sharded input on (0, 0), {g}", SHARDED_IN, m) for g, m in GRIDS]

device = ttnn.open_device(device_id=0)
for case, (name, in_mem, out_mem) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    x = ttnn.from_torch(eighths((1, 1, USERS, W)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
    xt = ttnn.to_torch(x).float()[0, 0]  # [users, W]
    want = [xt[:, :NQ * HD].reshape(1, USERS, NQ, HD),
            xt[:, NQ * HD:(NQ + NKV) * HD].reshape(1, USERS, NKV, HD),
            xt[:, (NQ + NKV) * HD:].reshape(1, USERS, NKV, HD)]
    try:
        got = ttnn.experimental.nlp_create_qkv_heads_decode(x, num_heads=NQ, num_kv_heads=NKV, memory_config=out_mem)
        wrong = sum(int((ttnn.to_torch(g).float() != w).sum()) for g, w in zip(got, want))
        total = sum(w.numel() for w in want)
        print(f"nlp_create_qkv_heads_decode, {name}: {wrong} of {total} values wrong", flush=True)
    except RuntimeError as e:
        print(f"nlp_create_qkv_heads_decode, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
