"""ttnn.experimental.nlp_kv_cache_load_slice into a preallocated HEIGHT-sharded output not on the walked grid.

The factory (nlp_kv_cache_load_slice_program_factory.cpp) gives head i (fused batch * heads) to
core {i % W, i / W}, W = width of the output grid's FIRST range: a row-major walk from (0, 0). The output buffer
places head i on corerange_to_cores(output grid, n, orientation == ROW_MAJOR)[i]. The default output grid
num_cores_to_corerangeset(heads, device grid, true) (device op) matches the walk, but create_output_tensors
returns a preallocated `output_tensor` unchanged and validation never looks at it. Input is DRAM
interleaved, as in the op's test. Values are multiples of 1/8 and the op only moves data: exact comparison."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


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


# (name, input shape, seq_len_start, seq_len_end, preallocated output (grid, orientation) or None)
CASES = [
    ("[1, 4, 64, 64], seq 0..32, default output, control", (1, 4, 64, 64), 0, 32, None),
    ("[9, 3, 64, 64], seq 32..64, default output (test config), control", (9, 3, 64, 64), 32, 64, None),
    ("[1, 4, 64, 64], seq 0..32, preallocated on (0, 0)-(3, 0), control", (1, 4, 64, 64), 0, 32,
     (grid(((0, 0), (3, 0))), RM)),
    ("[1, 4, 64, 64], seq 0..32, preallocated on 2x2 at (0, 0), control", (1, 4, 64, 64), 0, 32,
     (grid(((0, 0), (1, 1))), RM)),
    ("[1, 4, 64, 64], seq 0..32, preallocated on (1, 0)-(4, 0)", (1, 4, 64, 64), 0, 32,
     (grid(((1, 0), (4, 0))), RM)),
    ("[1, 4, 64, 64], seq 0..32, preallocated on 2x2 at (0, 0), COL_MAJOR", (1, 4, 64, 64), 0, 32,
     (grid(((0, 0), (1, 1))), CM)),
    ("[1, 4, 64, 64], seq 0..32, preallocated on (0, 1)-(1, 1) then (0, 0)-(1, 0)", (1, 4, 64, 64), 0, 32,
     (grid(((0, 1), (1, 1)), ((0, 0), (1, 0))), RM)),
]

device = ttnn.open_device(device_id=0)
for case, (name, shape, start, end, prealloc) in enumerate(CASES):
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    x = ttnn.from_torch(eighths(shape), layout=ttnn.TILE_LAYOUT, device=device)  # DRAM interleaved
    want = ttnn.to_torch(x).float()[:, :, start:end, :]
    out = None
    if prealloc is not None:
        mem = ttnn.MemoryConfig(HEIGHT, ttnn.BufferType.L1, ttnn.ShardSpec(prealloc[0], [end - start, shape[3]],
                                                                            prealloc[1]))
        out = ttnn.from_torch(torch.full(tuple(want.shape), 100.0, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                              device=device, memory_config=mem)
    try:
        y = ttnn.experimental.nlp_kv_cache_load_slice(x, seq_len_start=start, seq_len_end=end, output_tensor=out)
        wrong = int((ttnn.to_torch(y).float() != want).sum())
        print(f"nlp_kv_cache_load_slice, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"nlp_kv_cache_load_slice, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
