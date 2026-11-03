"""ttnn.experimental.rotary_embedding_llama with HEIGHT-sharded cos/sin whose shard order is not the walked one.

Prefill (is_decode_mode=False, interleaved input, sharded cos/sin): the prefill-sharded factory gives sequence tile i
to `grid_to_cores(n, grid_x, grid_y, true)[i]`, a row-major walk of the whole compute grid from (0, 0)
(rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp). On the fast path
(cos/sin on exactly all cores, ROW_MAJOR, batch 1, one tile per core) each core multiplies with its
own resident cos/sin shard (borrowed DFB), but buffers place cos shard i on
corerange_to_cores(cos grid, n, true)[i], range by range. Nothing requires one range or a range order.
Decode (is_decode_mode=True): the sharded factory runs on the input grid's bounding box (sharded factory)
and borrows input, cos, sin, trans_mat and output from each core's own shards, so core X pairs input user
idx_in(X) with cos user idx_cos(X); validation (device op) never compares the cos/sin grid or orientation
with the input's. The op computes out = x * cos + (x @ T) * sin, T the 32x32 pair-rotation matrix applied per
32-column tile (decode broadcasts row 0 of each user's cos/sin tile over the head rows). Data: x in multiples of 1/8 in [-2, 2], cos/sin in multiples of 1/4 in [-1, 1], so every product
and the sum are exact in bfloat16 (at most 8 significant bits). Tolerance: a value counts as wrong if it is off
by more than 1/64, half a bfloat16 ulp for |out| < 4, which also covers one rounding if the hardware rounds."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
L1 = ttnn.BufferType.L1
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
HD, TOL = 128, 1 / 64


def grid(*ranges):  # a list keeps the range order as written
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def rand_steps(shape, step, hi):  # multiples of `step` in [-hi, hi]
    n = round(hi / step)
    return (torch.randint(-n, n + 1, shape) * step).to(torch.bfloat16)


def err_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None)
    if line is None:
        i = next((j for j, l in enumerate(lines) if l == "info:"), None)
        line = lines[i + 1] if i is not None and i + 1 < len(lines) else lines[0]
    return re.sub(r"/\S*/", "", line)[:200]


def trans_mat(rows=32):  # tt_py_test_utils_common.tensor_utils.get_rot_transformation_mat, repeated down the rows
    t = torch.zeros(32, 32)
    t[torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    t[torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    return t.repeat(rows // 32, 1)[None, None].to(torch.bfloat16)


def reference(x, cos, sin, t):
    rot = (x.reshape(*x.shape[:-1], HD // 32, 32) @ t).reshape(x.shape)
    return x * cos + rot * sin


device = ttnn.open_device(device_id=0)
gx, gy = device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y
n_all, hx, hy = gx * gy, gx // 2, gy // 2
FULL = grid(((0, 0), (gx - 1, gy - 1)))
# Prefill: (name, cos/sin memory config or None for DRAM interleaved)
PREFILL = [
    ("prefill, cos/sin interleaved, control", None),
    (f"prefill, cos/sin on all {n_all} cores as one range, control", FULL),
    (f"prefill, cos/sin on (0, 0)-({hx - 1}, {gy - 1}) + ({hx}, 0)-({gx - 1}, {gy - 1})",
     grid(((0, 0), (hx - 1, gy - 1)), ((hx, 0), (gx - 1, gy - 1)))),
    (f"prefill, cos/sin on (0, {hy})-({gx - 1}, {gy - 1}) then (0, 0)-({gx - 1}, {hy - 1})",
     grid(((0, hy), (gx - 1, gy - 1)), ((0, 0), (gx - 1, hy - 1)))),
]
# Decode: (name, input/trans_mat grid, cos/sin grid, cos/sin orientation)
SQ8 = grid(((0, 0), (3, 1)))
DECODE = [
    ("decode, input and cos/sin on 4x2 at (0, 0), control", SQ8, SQ8, RM),
    ("decode, input on (0, 0)-(7, 0), cos/sin COL_MAJOR there, control", grid(((0, 0), (7, 0))),
     grid(((0, 0), (7, 0))), CM),
    ("decode, input on 4x2 at (0, 0), cos/sin COL_MAJOR there", SQ8, SQ8, CM),
    ("decode, input on 4x2 at (0, 0), cos/sin on (0, 0)-(7, 0)", SQ8, grid(((0, 0), (7, 0))), RM),
]
t32 = trans_mat()[0, 0].float()
case = 0
for name, cs_grid in PREFILL:
    torch.manual_seed(case)  # new data per case, so stale L1 from an earlier case cannot match by chance
    case += 1
    seq, rows = 512, 32 * n_all  # 16 sequence tiles; cos/sin one tile per core
    put = lambda h, mem=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=device,
                                                               memory_config=mem)
    cs_mem = ttnn.DRAM_MEMORY_CONFIG if cs_grid is None else ttnn.MemoryConfig(
        HEIGHT, L1, ttnn.ShardSpec(cs_grid, [32, HD], RM))
    x = put(rand_steps((1, 2, seq, HD), 1 / 8, 2))
    cos, sin = put(rand_steps((1, 1, rows, HD), 1 / 4, 1), cs_mem), put(rand_steps((1, 1, rows, HD), 1 / 4, 1), cs_mem)
    t = put(trans_mat())
    want = reference(ttnn.to_torch(x).float(), ttnn.to_torch(cos).float()[:, :, :seq],
                     ttnn.to_torch(sin).float()[:, :, :seq], ttnn.to_torch(t).float()[0, 0])
    try:
        y = ttnn.experimental.rotary_embedding_llama(x, cos, sin, t, is_decode_mode=False)
        wrong = int(((ttnn.to_torch(y).float() - want).abs() > TOL).sum())
        print(f"rotary_embedding_llama, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"rotary_embedding_llama, {name}: {err_line(e)}", flush=True)
for name, in_grid, cs_grid, cs_orient in DECODE:
    torch.manual_seed(case)
    case += 1
    users = in_grid.num_cores()
    in_mem = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(in_grid, [32, HD], RM))
    cs_mem = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(cs_grid, [32, HD], cs_orient))
    t_mem = ttnn.MemoryConfig(HEIGHT, L1, ttnn.ShardSpec(in_grid, [32, 32], RM))
    put = lambda h, mem: ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    x = put(rand_steps((1, users, 32, HD), 1 / 8, 2), in_mem)  # [1, users, heads, head_dim]
    # One cos/sin row per user, repeated over the 32 head rows as in decode RoPE; the decode kernel
    # broadcasts row 0 of the cos/sin tile over the rows (BroadcastDim::Row), so the reference does too.
    cos = put(rand_steps((1, users, 1, HD), 1 / 4, 1).repeat(1, 1, 32, 1), cs_mem)
    sin = put(rand_steps((1, users, 1, HD), 1 / 4, 1).repeat(1, 1, 32, 1), cs_mem)
    t = put(trans_mat(32 * users), t_mem)
    want = reference(ttnn.to_torch(x).float(), ttnn.to_torch(cos).float()[:, :, :1],
                     ttnn.to_torch(sin).float()[:, :, :1], ttnn.to_torch(t).float()[0, 0, :32])
    try:
        y = ttnn.experimental.rotary_embedding_llama(x, cos, sin, t, is_decode_mode=True)
        wrong = int(((ttnn.to_torch(y).float() - want).abs() > TOL).sum())
        print(f"rotary_embedding_llama, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"rotary_embedding_llama, {name}: {err_line(e)}", flush=True)
ttnn.close_device(device)
