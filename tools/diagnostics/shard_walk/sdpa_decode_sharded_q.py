"""ttnn.transformer.scaled_dot_product_attention_decode with Q HEIGHT_SHARDED in L1 on grids other than the walked one.

Without sub_core_grids, with a sharded Q or output and no column-major group indexing, sdpa_decode_program_factory.cpp puts the reducer/output core of batch b at {b % grid_size.x, b / grid_size.x} of the program grid, an
origin-anchored row-major walk. The reader takes batch b's Q from that core's L1 at the Q buffer address
(kernels/dataflow/dataflow_common.hpp read_q: "Q is sharded - read from output core's L1"), and a sharded output is
written there too (its CB is backed by the output buffer). Buffers place shard b on corerange_to_cores(grid, B,
row_major)[b]. Validation checks layouts only (sdpa_decode_device_operation.cpp; its Q grid check is commented out). One Blackhole chip or ttsim.

B=4, 32 query heads, 1 KV head, S=128, D=64, Q shard [32, 64], program grid 4x2 (two cores per batch, output cores
(b, 0)), cur_pos [127, 95, 64, 40]. Q, K, V are bf16 multiples of 1/8 in [-2, 2], [-1, 1], [-2, 2], with a new seed per
case so stale L1 from an earlier case cannot pass. Expected: torch softmax(scale * Q K^T) V over positions 0..cur_pos,
from ttnn.to_torch of the device inputs. Not exact (bf16, exp), so a value is wrong if |got - want| > 0.05 +
0.03 * |want| (or NaN); HiFi4, fp32 dest, exp_approx_mode False."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
B, NH, S, D = 4, 32, 128, 64
CUR_POS = [127, 95, 64, 40]
SCALE = D**-0.5


def grid(*ranges):  # a list keeps the ranges in the order given
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def eighths(lo, hi, shape):
    return (torch.randint(8 * lo, 8 * hi + 1, shape) / 8).to(torch.bfloat16)


def error_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None)
    if line is None:
        line = lines[lines.index("info:") + 1] if "info:" in lines[:-1] else lines[0]
    return re.sub(r"\S*/", "", line)[:170]


ROW0, ROW1 = grid(((0, 0), (3, 0))), grid(((0, 1), (3, 1)))
CASES = [  # (name, Q shard grid, orientation, output sharded on the same grid)
    ("Q on (0, 0)-(3, 0), control", ROW0, RM, False),
    ("Q and output on (0, 0)-(3, 0), control", ROW0, RM, True),
    ("Q on (0, 1)-(3, 1)", ROW1, RM, False),
    ("Q on (1, 0)-(4, 0)", grid(((1, 0), (4, 0))), RM, False),
    ("Q on 2x2 at (0, 0)", grid(((0, 0), (1, 1))), RM, False),
    ("Q on 2x2 at (0, 0), COL_MAJOR", grid(((0, 0), (1, 1))), CM, False),
    ("Q and output on (0, 1)-(3, 1)", ROW1, RM, True),
]

device = ttnn.open_device(device_id=0)
program_config = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=CC(4, 2), q_chunk_size=32, k_chunk_size=32, exp_approx_mode=False
)
ck = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    dst_full_sync_en=False,
)
for seed, (name, shard_grid, orientation, out_sharded) in enumerate(CASES):
    torch.manual_seed(seed)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(shard_grid, [32, D], orientation)
    )
    tq = ttnn.from_torch(eighths(-2, 2, (1, B, NH, D)), layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
    tk = ttnn.from_torch(eighths(-1, 1, (B, 1, S, D)), layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(eighths(-2, 2, (B, 1, S, D)), layout=ttnn.TILE_LAYOUT, device=device)
    q, k, v = (ttnn.to_torch(t).float() for t in (tq, tk, tv))
    want = torch.empty(1, B, NH, D)
    for b, p in enumerate(CUR_POS):
        want[0, b] = torch.softmax(SCALE * q[0, b] @ k[b, 0, : p + 1].T, -1) @ v[b, 0, : p + 1]
    try:
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tq,
            tk,
            tv,
            cur_pos=CUR_POS,
            scale=SCALE,
            program_config=program_config,
            compute_kernel_config=ck,
            memory_config=mem if out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out).float()[:, :, :NH]
        wrong = int((~((got - want).abs() <= 0.05 + 0.03 * want.abs())).sum())
        print(f"sdpa_decode, {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"sdpa_decode, {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
