"""ttnn.experimental.matmul_decode (full, partial_width_sharded=True, and batched rank-4 A) with WIDTH_SHARDED A or B
in COL_MAJOR order, B on an offset grid, or an output grid listed in another order.

All three factories walk the A grid row-major for the K-slice gather: sender_cores = corerange_to_cores(A grid,
nullopt, true), sender id = gathered slot (full_width_sharded_program_factory.cpp,
partial_width_sharded_program_factory.cpp, batched_width_sharded_program_factory.cpp). Partial and
batched walk the B grid row-major too (partial, batched) and take B core idx to hold B shard idx: partial
k_idx = idx / N_blocks, n_idx = idx % N_blocks, base core = b_cores[n_idx], which reduces into its own output shard
(partial); batched writes output block (idx / n_blocks, idx % n_blocks) (batched).
Full multiplies each core's own B shard into its own output shard; the output is ROW_MAJOR on B's grid (device op). Validation checks counts and shard shapes only (matmul_decode_device_operation.cpp); partial checks
that each base core is in the output grid (factory). Buffers place shard i on corerange_to_cores(grid, n,
orientation == ROW_MAJOR)[i]. One Blackhole chip or ttsim. A holds multiples of 1/8 in [-2, 2]; each column of
(each batch of) B has one nonzero (1, -1, 2 or 1/2), so every output is one product plus zeros: exact in bfloat16
(the factories use HiFi4). The full and partial readers multicast a semaphore increment (inc_multicast), which ttsim
v1.15.1 does not implement (it aborts the run), so the batched cases run first and the full/partial cases last; the
partial case on an offset B grid stops on the host before any kernel runs."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
M, K = 32, 128


def grid(*ranges):
    return ttnn.CoreRangeSet([ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges])


def width_sharded(shard_grid, shard, orientation=RM):
    return ttnn.MemoryConfig(WIDTH, ttnn.BufferType.L1, ttnn.ShardSpec(shard_grid, list(shard), orientation))


def error_line(e):  # the "Kernel ..." line, else the message after "info:", else the first line
    lines = [l.strip() for l in str(e).splitlines() if l.strip()]
    if "info:" in lines[:-1]:
        lines.insert(0, lines[lines.index("info:") + 1])
    return next((l for l in lines if l.startswith("Kernel")), lines[0])[:170]


def selection(batch, k, n):  # batch u, column j: one nonzero, in row (37 j + 11 u + 5) % k
    b = torch.zeros(batch, k, n)
    cols = torch.arange(n)
    for u in range(batch):
        b[u, (cols * 37 + 11 * u + 5) % k, cols] = torch.tensor([1.0, -1.0, 2.0, 0.5]).repeat(n)[:n]
    return b.to(torch.bfloat16)


# kind: (A shape, A shard, B as stored on device, B shard, unfold(device B) -> [.., K, N], partial_width_sharded)
KINDS = {
    "full": ((M, K), (M, 32), selection(1, K, 128)[0], (K, 32), lambda t: t, False),
    # partial: K_blocks = 2 (Kc = 64), N_blocks = 2 (Nc = 32); B[kb*64 + r, n] stored at [r, kb*64 + n]
    "partial": ((M, K), (M, 32), selection(1, K, 64)[0].reshape(2, 64, 64).permute(1, 0, 2).reshape(64, 128),
                (64, 32), lambda t: t.reshape(64, 2, 64).permute(1, 0, 2).reshape(K, 64), True),
    # batched: batch 2 = b_blocks (Bc = 1), n_blocks = 2 (Nc = 32); B[b, r, n] stored at [0, 0, r, b*64 + n]
    "batched": ((1, 2, M, K), (2 * M, 32), selection(2, K, 64).reshape(2, 1, K, 64).permute(1, 2, 0, 3).reshape(
        1, 1, K, 128), (K, 32), lambda t: t.reshape(K, 2, 64).permute(1, 0, 2), False),
}
SQ = grid(((0, 0), (1, 1)))
CASES = [  # (name, kind, A orientation, B grid, B orientation, output memory config or None for the default)
    ("A and B ROW_MAJOR on 2x2 at (0, 0), control", "batched", RM, SQ, RM, None),
    ("A COL_MAJOR on 2x2 at (0, 0)", "batched", CM, SQ, RM, None),
    ("B COL_MAJOR on 2x2 at (0, 0)", "batched", RM, SQ, CM, None),
    ("B on 2x2 at (0, 1), default output", "partial", RM, grid(((0, 1), (1, 2))), RM, None),
    # the cases below need multicast atomics (not on ttsim v1.15.1)
    ("A and B ROW_MAJOR on 2x2 at (0, 0), control", "full", RM, SQ, RM, None),
    ("A COL_MAJOR on 2x2 at (0, 0)", "full", CM, SQ, RM, None),
    ("B COL_MAJOR on 2x2 at (0, 0)", "full", RM, SQ, CM, None),
    ("A and B ROW_MAJOR on 2x2 at (0, 0), control", "partial", RM, SQ, RM, None),
    ("A COL_MAJOR on 2x2 at (0, 0)", "partial", CM, SQ, RM, None),
    ("B COL_MAJOR on 2x2 at (0, 0)", "partial", RM, SQ, CM, None),
    ("output on cores (1, 0) then (0, 0)", "partial", RM, SQ, RM,
     width_sharded(grid(((1, 0), (1, 0)), ((0, 0), (0, 0))), (M, 32))),
]

device = ttnn.open_device(device_id=0)
for name, kind, a_orientation, b_grid, b_orientation, out_mem in CASES:
    torch.manual_seed(0)
    a_shape, a_shard, b_host, b_shard, unfold, partial = KINDS[kind]
    a_host = (torch.randint(-16, 17, a_shape) / 8).to(torch.bfloat16)
    a = ttnn.from_torch(a_host, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=width_sharded(SQ, a_shard, a_orientation))
    b = ttnn.from_torch(b_host, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=width_sharded(b_grid, b_shard, b_orientation))
    b_full = unfold(ttnn.to_torch(b).float())
    want = (ttnn.to_torch(a).float().reshape(-1, M, K) @ b_full).reshape(*a_shape[:-1], b_full.shape[-1])
    try:
        y = ttnn.experimental.matmul_decode(a, b, partial_width_sharded=partial, output_mem_config=out_mem)
        wrong = int((ttnn.to_torch(y).float().reshape(want.shape) != want).sum())
        print(f"matmul_decode ({kind}), {name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        print(f"matmul_decode ({kind}), {name}: {error_line(e)}", flush=True)
ttnn.close_device(device)
