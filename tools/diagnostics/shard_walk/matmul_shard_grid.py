"""ttnn.matmul / ttnn.linear with sharded operands whose shard grid is not where the matmul factories look for it.

- Auto config (matmul_program_config.cpp): for a BLOCK_SHARDED A it keeps only the size of the shard
  grid's bounding box; normalize_program_config anchors that rectangle at (0, 0) (matmul_program_config_types.hpp).
- 2D mcast factory (matmul_multicore_reuse_mcast_2d_program_factory.cpp): the work cores are the rectangle from allowed_worker_cores' start, (0, 0) by default; every core reads
  its in0 shard in place (borrowed DFB) as if it held shard (y - start_y, x - start_x); a HEIGHT_SHARDED in0 is read
  in place on the start column. Validation checks only that A lies in the device grid (matmul_device_operation.cpp).
- 1D mcast factory, mcast_in0 / mcast_in1 (matmul_multicore_reuse_mcast_1d_program_factory.cpp):
  block i goes to the i-th row-major core of the compute_with_storage_grid_size rectangle at (0, 0); a user
  BLOCK_SHARDED 1-row/1-column output grid is kept as given (matmul_device_operation.cpp) and only checked
  to lie inside that rectangle.
- 1D gather_in0 (same file): ring core i is the i-th row-major A core; it reads N-block i of a
  DRAM-interleaved B and writes it into its own output shard. The output orientation is not checked.
Buffers place shard i on corerange_to_cores(grid, n, orientation == ROW_MAJOR)[i] (HEIGHT/WIDTH) or by position from
the start of the single range (BLOCK). One Blackhole chip or ttsim. A holds multiples of 1/8 in [-2, 2]; each column
of B has one nonzero (1, -1, 2 or 1/2), so every output is one product plus zeros: exact in bfloat16 (HiFi4)."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
WIDTH, HEIGHT, BLOCK = (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        ttnn.TensorMemoryLayout.BLOCK_SHARDED)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def grid(*ranges):
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def sharded(layout, shard_grid, shard, orientation=RM):
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(shard_grid, list(shard), orientation))


def eighths(shape):
    return (torch.randint(-16, 17, shape) / 8).to(torch.bfloat16)


def selection(k, n):  # column j of B has one nonzero, in row (37 j + 5) % k
    b = torch.zeros(1, 1, k, n)
    cols = torch.arange(n)
    b[0, 0, (cols * 37 + 5) % k, cols] = torch.tensor([1.0, -1.0, 2.0, 0.5]).repeat(n)[:n]
    return b.to(torch.bfloat16)


def error_line(e):  # the "Kernel ..." line, else the message after "info:", else the first line
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    if "info:" in lines[:-1]:
        lines.insert(0, lines[lines.index("info:") + 1])
    return next((l for l in lines if l.startswith("Kernel")), lines[0])[:170]


def mcast_2d(grid_xy, in0_block_w, per_core_m, per_core_n):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=CC(*grid_xy), in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=per_core_m, per_core_N=per_core_n, transpose_mcast=False, fused_activation=None)


def mcast_1d(grid_xy, in0_block_w, per_core_m, per_core_n, mcast_in0, gather_in0=False):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=CC(*grid_xy), in0_block_w=in0_block_w, out_subblock_h=1, out_subblock_w=1,
        per_core_M=per_core_m, per_core_N=per_core_n, fuse_batch=True, fused_activation=None, mcast_in0=mcast_in0,
        gather_in0=gather_in0)


SQ00, SQ11 = grid(((0, 0), (1, 1))), grid(((1, 1), (2, 2)))
# (name, op, A shape, A memory config, N, output memory config, program config); B is [1, 1, K, N] in DRAM
CASES = [
    ("auto config, A BLOCK_SHARDED 2x2 at (0, 0), control", "matmul", (1, 1, 64, 128),
     sharded(BLOCK, SQ00, (32, 64)), 64, DRAM, None),
    ("auto config, A BLOCK_SHARDED 2x2 at (0, 0), control", "linear", (1, 1, 64, 128),
     sharded(BLOCK, SQ00, (32, 64)), 64, DRAM, None),
    ("2D config (2, 4), A HEIGHT_SHARDED on (0, 0)-(0, 3), control", "matmul", (1, 1, 128, 64),
     sharded(HEIGHT, grid(((0, 0), (0, 3))), (32, 64)), 64, DRAM, mcast_2d((2, 4), 2, 1, 1)),
    ("1D mcast_in1 config (1, 4), output BLOCK_SHARDED on (0, 0)-(0, 3), control", "matmul", (1, 1, 128, 64), DRAM,
     64, sharded(BLOCK, grid(((0, 0), (0, 3))), (32, 64)), mcast_1d((1, 4), 2, 1, 2, False)),
    ("1D mcast_in0 config (8, 8), output BLOCK_SHARDED on (0, 0)-(3, 0), control", "matmul", (1, 1, 32, 128), DRAM,
     128, sharded(BLOCK, grid(((0, 0), (3, 0))), (32, 32)), mcast_1d((8, 8), 2, 1, 1, True)),
    ("1D gather_in0, A and output WIDTH_SHARDED 2x2 at (0, 0), control", "matmul", (1, 1, 32, 128),
     sharded(WIDTH, SQ00, (32, 32)), 128, sharded(WIDTH, SQ00, (32, 32)), mcast_1d((2, 2), 1, 1, 1, False, True)),
    ("auto config, A BLOCK_SHARDED 2x2 at (1, 1)", "matmul", (1, 1, 64, 128),
     sharded(BLOCK, SQ11, (32, 64)), 64, DRAM, None),
    ("auto config, A BLOCK_SHARDED 2x2 at (1, 1)", "linear", (1, 1, 64, 128),
     sharded(BLOCK, SQ11, (32, 64)), 64, DRAM, None),
    ("2D config (2, 4), A HEIGHT_SHARDED on (3, 0)-(3, 3)", "matmul", (1, 1, 128, 64),
     sharded(HEIGHT, grid(((3, 0), (3, 3))), (32, 64)), 64, DRAM, mcast_2d((2, 4), 2, 1, 1)),
    ("1D gather_in0, A WIDTH_SHARDED 2x2 at (0, 0), output COL_MAJOR", "matmul", (1, 1, 32, 128),
     sharded(WIDTH, SQ00, (32, 32)), 128, sharded(WIDTH, SQ00, (32, 32), CM), mcast_1d((2, 2), 1, 1, 1, False, True)),
    ("1D mcast_in1 config (8, 8), output BLOCK_SHARDED on (0, 0)-(0, 3)", "matmul", (1, 1, 128, 64), DRAM, 64,
     sharded(BLOCK, grid(((0, 0), (0, 3))), (32, 64)), mcast_1d((8, 8), 2, 1, 2, False)),
    ("1D mcast_in0 config (8, 8), output BLOCK_SHARDED on (0, 2)-(3, 2)", "matmul", (1, 1, 32, 128), DRAM, 128,
     sharded(BLOCK, grid(((0, 2), (3, 2))), (32, 32)), mcast_1d((8, 8), 2, 1, 1, True)),
]

device = ttnn.open_device(device_id=0)
compute = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=False)
for name, op_name, a_shape, a_mem, n, out_mem, config in CASES:
    torch.manual_seed(0)
    a = ttnn.from_torch(eighths(a_shape), layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)
    b = ttnn.from_torch(selection(a_shape[-1], n), layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    want = ttnn.to_torch(a).float() @ ttnn.to_torch(b).float()
    op = ttnn.matmul if op_name == "matmul" else ttnn.linear
    try:
        y = op(a, b, memory_config=out_mem, program_config=config, compute_kernel_config=compute)
        wrong = int((ttnn.to_torch(y).float() != want).sum())
        print(f"ttnn.{op_name}, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"ttnn.{op_name}, {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
