"""ttnn.experimental.nlp_concat_heads_decode with a HEIGHT-sharded input or a preallocated output not on the walked grid.

The input [1, users, 32, head_dim] holds user u's heads on one core; buffers place user u on
corerange_to_cores(input grid, n, orientation == ROW_MAJOR)[u]. The output [1, 1, 32, heads * head_dim] is
WIDTH-sharded, head h on corerange_to_cores(output grid, n, orientation == ROW_MAJOR)[h]. The default factory
(chosen for one input range at (0, 0) and no sub_core_grids, prim) gives head h to
`grid_to_cores(n, out_bbox.end.x + 1, out_bbox.end.y + 1, true)[h]` (program factory) and its
reader walks the input cores x-fastest from (0, 0) (NoC tables; reader kernel), assuming user u is on
row-major core u. Validation (device op) checks neither the input orientation nor a preallocated output's
grid or orientation; compute_output_specs returns the preallocated spec unchanged. The last two cases
use the sibling subcoregrids factory (input off (0, 0), sub_core_grids given), which also walks the input
row-major (subcoregrids factory). Values are multiples of 1/8 and the op only moves data: exact comparison."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
L1 = ttnn.BufferType.L1
WIDTH, HEIGHT = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED
HD, USERS = 128, 4


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


ROW4, SQ = grid(((0, 0), (3, 0))), grid(((0, 0), (1, 1)))
# (name, input grid, input orientation, num_heads, preallocated output (grid, orientation) or None, sub_core_grids)
CASES = [
    ("input on (0, 0)-(3, 0), control", ROW4, RM, 8, None, None),
    ("input on 2x2 at (0, 0), control", SQ, RM, 8, None, None),
    ("input on (0, 0)-(3, 0), COL_MAJOR, control", ROW4, CM, 8, None, None),
    ("input on 2x2 at (0, 0), COL_MAJOR", SQ, CM, 8, None, None),
    ("4 heads, preallocated output on (0, 0)-(3, 0), control", ROW4, RM, 4, (ROW4, RM), None),
    ("4 heads, preallocated output on (1, 0)-(4, 0)", ROW4, RM, 4, (grid(((1, 0), (4, 0))), RM), None),
    ("4 heads, preallocated output on 2x2 at (0, 0), COL_MAJOR", ROW4, RM, 4, (SQ, CM), None),
    ("subcoregrids factory, input on 2x2 at (1, 0), control", grid(((1, 0), (2, 1))), RM, 8, None,
     grid(((1, 0), (8, 0)))),
    ("subcoregrids factory, input on 2x2 at (1, 0), COL_MAJOR", grid(((1, 0), (2, 1))), CM, 8, None,
     grid(((1, 0), (8, 0)))),
]

device = ttnn.open_device(device_id=0)
for case, (name, in_grid, in_orient, heads, prealloc, sub_grids) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    in_mem = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(in_grid, [32, HD], in_orient))
    x = ttnn.from_torch(eighths((1, USERS, 32, HD)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
    xt = ttnn.to_torch(x).float()
    want = xt[0, :, :heads, :].reshape(USERS, heads * HD)  # row u = user u's heads side by side
    out = None
    if prealloc is not None:
        out_mem = ttnn.MemoryConfig(WIDTH, L1, ttnn.ShardSpec(prealloc[0], [32, HD], prealloc[1]))
        out = ttnn.from_torch(torch.full((1, 1, 32, heads * HD), 100.0, dtype=torch.bfloat16),
                              layout=ttnn.TILE_LAYOUT, device=device, memory_config=out_mem)
    try:
        y = ttnn.experimental.nlp_concat_heads_decode(x, num_heads=heads, output_tensor=out, sub_core_grids=sub_grids)
        got = ttnn.to_torch(y).float()[0, 0, :USERS, :]  # rows past the users are padding
        wrong = int((got != want).sum())
        print(f"nlp_concat_heads_decode, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"nlp_concat_heads_decode, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
