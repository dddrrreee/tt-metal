"""The decode flow nlp_create_qkv_heads_decode -> scaled_dot_product_attention_decode with default memory configs.

nlp_create_qkv_heads_decode with memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG (no shard spec) puts user b's Q
heads on row-major core b of the whole device grid. With a sharded Q, sdpa_decode_program_factory.cpp
places batch b's reducer, which reads batch b's Q from its own L1, at {b % grid_size.x, b / grid_size.x} of the
SDPAProgramConfig grid. The two agree only when the program grid is as wide as the device grid, or when every user
fits in its first row. An (8, 4) program grid is a common choice; on Blackhole the device grid is wider than 8, so
batch 16 or 32 breaks. Validation does not compare Q's grid with the program grid (the check at
sdpa_decode_device_operation.cpp is commented out). One Blackhole chip or ttsim.

NQ=8 query heads, 1 KV head, head_dim 128, S=256. Q, K, V are bf16 multiples of 1/8 in [-2, 2], [-1, 1], [-2, 2],
with a new seed per case. Expected: torch softmax(scale * Q K^T) V over positions 0..cur_pos[b], computed from
ttnn.to_torch of the device tensors. Not exact (bf16, exp): a value counts as wrong if |got - want| > 0.05 +
0.03 * |want| (or NaN); HiFi4, fp32 dest, exp_approx_mode False."""
import re

import torch
import ttnn

CC = ttnn.CoreCoord
NQ, NKV, HD, S = 8, 1, 128, 256
SCALE = HD**-0.5


def eighths(lo, hi, shape):
    return (torch.randint(8 * lo, 8 * hi + 1, shape) / 8).to(torch.bfloat16)


def error_line(e):
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    line = next((l for l in lines if l.startswith("Kernel")), None) or next(
        (l for l in lines if not l.startswith(("TT_FATAL @", "TT_THROW @", "info:", "backtrace"))), lines[0])
    return re.sub(r"\S*/", "", line)[:170]


device = ttnn.open_device(device_id=0)
dev_grid = device.compute_with_storage_grid_size()
FULL = (dev_grid.x, dev_grid.y)
CASES = [  # (name, batch, SDPAProgramConfig grid)
    ("batch 8, program grid (8, 4), control", 8, (8, 4)),
    (f"batch 16, program grid = device grid {FULL}, control", 16, FULL),
    ("batch 16, program grid (8, 4)", 16, (8, 4)),
    ("batch 32, program grid (8, 4)", 32, (8, 4)),
]
ck = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
    dst_full_sync_en=False,
)
print(f"device grid {FULL}", flush=True)
for seed, (name, batch, program_grid) in enumerate(CASES):
    torch.manual_seed(seed)
    cur_pos = [(37 * b + 40) % 216 + 40 for b in range(batch)]  # in [40, 255]
    fused = ttnn.from_torch(eighths(-2, 2, (1, 1, batch, (NQ + 2 * NKV) * HD)), layout=ttnn.TILE_LAYOUT,
                            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tk = ttnn.from_torch(eighths(-1, 1, (batch, NKV, S, HD)), layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(eighths(-2, 2, (batch, NKV, S, HD)), layout=ttnn.TILE_LAYOUT, device=device)
    try:
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused, num_heads=NQ, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        qt, k, v = ttnn.to_torch(q).float(), ttnn.to_torch(tk).float(), ttnn.to_torch(tv).float()
        want = torch.empty(1, batch, NQ, HD)
        for b, p in enumerate(cur_pos):
            want[0, b] = torch.softmax(SCALE * qt[0, b, :NQ] @ k[b, 0, : p + 1].T, -1) @ v[b, 0, : p + 1]
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            tk,
            tv,
            cur_pos=cur_pos,
            scale=SCALE,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=CC(*program_grid), q_chunk_size=32, k_chunk_size=32,
                exp_approx_mode=False
            ),
            compute_kernel_config=ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out).float()[:, :, :NQ]
        bad = ~((got - want).abs() <= 0.05 + 0.03 * want.abs())
        users = [b for b in range(batch) if bad[0, b].any()]
        print(f"sdpa_decode, {name}: {int(bad.sum())} of {want.numel()} values wrong; users wrong: {users}",
              flush=True)
    except RuntimeError as e:
        print(f"sdpa_decode, {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
