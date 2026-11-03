"""ttnn.experimental.group_attn_matmul with a HEIGHT_SHARDED A (q heads) or B (KV, one shard per user) on a shard
grid other than the first 32 cores of compute_with_storage_grid_size.

The factory puts q head i on split_work_to_cores(compute_with_storage_grid_size, 32, row_major) core i, i.e.
(i % 8, i / 8) for an (8, 8) grid (group_attn_matmul_program_factory.cpp), and reads the sharded
A and B in place through DFBs borrowed from the tensors on those cores. The in1 senders' NoC coordinates
are rebuilt from (x, 0) and (0, y). Validation checks the orientation and the core count of A and B, not
where their grids are (group_attn_matmul_device_operation.cpp). Buffers place shard i on
corerange_to_cores(grid, n, row_major)[i]. One Blackhole chip or ttsim. A holds multiples of 1/8 in [-2, 2]; each
column of each user's B has one nonzero (1, -1, 2 or 1/2), so every output is one product plus zeros: exact in
bfloat16 (HiFi4). Output in DRAM (the Python default)."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM = ttnn.ShardOrientation.ROW_MAJOR
HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
DRAM = ttnn.DRAM_MEMORY_CONFIG
Q_HEADS, USERS, HEAD_DIM, KV_LEN = 32, 32, 32, 32


def grid(*ranges):
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def error_line(e):  # the "Kernel ..." line, else the message after "info:", else the first line
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    if "info:" in lines[:-1]:
        lines.insert(0, lines[lines.index("info:") + 1])
    return next((l for l in lines if l.startswith("Kernel")), lines[0])[:170]


def selection():  # user u, column j of B: one nonzero, in row (37 j + 11 u + 5) % HEAD_DIM
    b = torch.zeros(USERS, 1, HEAD_DIM, KV_LEN)
    cols = torch.arange(KV_LEN)
    vals = torch.tensor([1.0, -1.0, 2.0, 0.5]).repeat(KV_LEN)[:KV_LEN]
    for u in range(USERS):
        b[u, 0, (cols * 37 + 11 * u + 5) % HEAD_DIM, cols] = vals
    return b.to(torch.bfloat16)


ORIGIN, OFFSET = grid(((0, 0), (7, 3))), grid(((0, 1), (7, 4)))
CASES = [  # (name, A shard grid or None for DRAM, B shard grid or None for DRAM)
    ("A on (0, 0)-(7, 3), control", ORIGIN, None),
    ("B on (0, 0)-(7, 3), control", None, ORIGIN),
    ("A on (0, 1)-(7, 4)", OFFSET, None),
    ("A on (0, 0)-(7, 1) and (0, 4)-(7, 5)", grid(((0, 0), (7, 1)), ((0, 4), (7, 5))), None),
    ("B on (0, 1)-(7, 4)", None, OFFSET),
]

device = ttnn.open_device(device_id=0)
compute = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=False)
for name, a_grid, b_grid in CASES:
    torch.manual_seed(0)
    a_mem = DRAM if a_grid is None else ttnn.MemoryConfig(
        HEIGHT, ttnn.BufferType.L1, ttnn.ShardSpec(a_grid, [USERS, HEAD_DIM], RM))
    b_mem = DRAM if b_grid is None else ttnn.MemoryConfig(
        HEIGHT, ttnn.BufferType.L1, ttnn.ShardSpec(b_grid, [HEAD_DIM, KV_LEN], RM))
    a_host = (torch.randint(-16, 17, (1, Q_HEADS, USERS, HEAD_DIM)) / 8).to(torch.bfloat16)
    a = ttnn.from_torch(a_host, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)
    b = ttnn.from_torch(selection(), layout=ttnn.TILE_LAYOUT, device=device, memory_config=b_mem)
    want = (ttnn.to_torch(a).float().transpose(0, 2) @ ttnn.to_torch(b).float()).transpose(0, 2)
    try:
        y = ttnn.experimental.group_attn_matmul(a, b, compute_with_storage_grid_size=CC(8, 8),
                                                memory_config=DRAM, compute_kernel_config=compute)
        wrong = int((ttnn.to_torch(y).float() != want).sum())
        print(f"group_attn_matmul, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"group_attn_matmul, {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
