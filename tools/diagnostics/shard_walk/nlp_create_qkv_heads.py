"""ttnn.experimental.nlp_create_qkv_heads with a WIDTH-sharded fused QKV input on shard grids other than the walked one.

The Sharded factory (nlp_create_qkv_heads_program_factory.cpp) assigns output head i to
`grid_to_cores(n, q_bbox.end.x + 1, q_bbox.end.y + 1, true)[i]`, a row-major walk from core (0, 0) over the Q
output grid's bounding box, and finds the input shard j holding a head at logical core (j % Wq, j / Wq) with
Wq = q_bbox.end.x + 1 (NoC tables from {x, 0} / {0, y}). Buffers place
input shard j on corerange_to_cores(input grid, n, true)[j] and output head i on
corerange_to_cores(q grid, n, true)[i]. The default Q grid is num_cores_to_corerangeset(nq, device grid, true)
(device op); a preallocated `output_tensors` triple keeps its own grid. Validation (device op) checks
neither the input grid's start, its ranges, nor its width against the Q grid. Values are multiples of 1/8, and
the op only moves data, so the comparison is exact."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM = ttnn.ShardOrientation.ROW_MAJOR
L1 = ttnn.BufferType.L1
WIDTH, HEIGHT = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED


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


# (name, num_q_heads, num_kv_heads, head_dim, input shard grid, preallocated output grid or None)
CASES = [
    ("8 heads, input on cores (0, 0)-(7, 0), control", 8, 8, 32, grid(((0, 0), (7, 0))), None),
    ("32 q / 2 kv heads, input on cores (0, 0)-(1, 0) (test config), control", 32, 2, 64, grid(((0, 0), (1, 0))), None),
    ("8 heads, input on (0, 0)-(7, 0), preallocated q/k/v on (0, 0)-(7, 0), control", 8, 8, 32,
     grid(((0, 0), (7, 0))), grid(((0, 0), (7, 0)))),
    ("2 heads, input on cores (1, 0)-(2, 0)", 2, 2, 32, grid(((1, 0), (2, 0))), None),
    ("8 heads, input on 4x2 at (0, 0)", 8, 8, 32, grid(((0, 0), (3, 1))), None),
    ("8 heads, input on cores (0, 0)-(3, 0) and (5, 0)-(8, 0)", 8, 8, 32, grid(((0, 0), (3, 0)), ((5, 0), (8, 0))), None),
    ("8 heads, input on (0, 0)-(7, 0), preallocated q/k/v on (1, 0)-(8, 0)", 8, 8, 32,
     grid(((0, 0), (7, 0))), grid(((1, 0), (8, 0)))),
]

device = ttnn.open_device(device_id=0)
for case, (name, nq, nkv, hd, in_grid, out_grid) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    n_in = in_grid.num_cores()
    qpc, kpc = nq // n_in, nkv // n_in  # heads per input shard
    per = (qpc + 2 * kpc) * hd  # shard width: [q heads, k heads, v heads] of one group
    in_mem = ttnn.MemoryConfig(WIDTH, L1, ttnn.ShardSpec(in_grid, [32, per], RM))
    x = ttnn.from_torch(eighths((1, 1, 32, per * n_in)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
    xt = ttnn.to_torch(x).float()

    def head(h, per_group, offset):
        col = (h // per_group) * per + offset + (h % per_group) * hd
        return xt[0, 0, :, col:col + hd]

    want = [torch.stack([head(h, qpc, 0) for h in range(nq)])[None],
            torch.stack([head(h, kpc, qpc * hd) for h in range(nkv)])[None],
            torch.stack([head(h, kpc, (qpc + kpc) * hd) for h in range(nkv)])[None]]
    outs = None
    if out_grid is not None:
        def prealloc(n):
            mem = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(out_grid, [32, hd], RM))
            return ttnn.from_torch(torch.full((1, n, 32, hd), 100.0, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=mem)
        outs = [prealloc(nq), prealloc(nkv), prealloc(nkv)]
    try:
        got = ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False,
            memory_config=ttnn.MemoryConfig(HEIGHT, L1), output_tensors=outs)
        wrong = sum(int((ttnn.to_torch(g).float() != w).sum()) for g, w in zip(got, want))
        total = sum(w.numel() for w in want)
        print(f"nlp_create_qkv_heads, {name}: {wrong} of {total} values wrong", flush=True)
    except RuntimeError as e:
        print(f"nlp_create_qkv_heads, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
