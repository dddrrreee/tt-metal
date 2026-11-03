"""ttnn.add / ttnn.where with HEIGHT_SHARDED inputs and a sharded output on the same grid in the other orientation.

binary_ng and ternary walk the cores in the first input's orientation (binary_ng_program_factory.cpp,
ternary_program_factory.cpp) and alias each core's output CB to the output buffer, so core k writes input shard k
into whatever output shard sits on core k. Their native-sharding checks compare only the grids, not the orientation
(binary_ng_utils.cpp, ternary_op_utils.cpp). One Blackhole chip or ttsim. Inputs are multiples of 1/8,
so a + b and where(p, t, f) are exact in bfloat16."""
import torch
import ttnn

CC = ttnn.CoreCoord
RM, CM = ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR
GRID = ttnn.CoreRangeSet({ttnn.CoreRange(CC(0, 0), CC(1, 1))})  # 2x2, the smallest grid where RM and CM differ


def mem(orientation):
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                             ttnn.ShardSpec(GRID, [32, 32], orientation))


device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
shape = (1, 1, 128, 32)  # four [32, 32] shards
a_h, b_h = ((torch.randint(-16, 17, shape) / 8).to(torch.bfloat16) for _ in range(2))
p_h = torch.randint(0, 2, shape).to(torch.bfloat16)
put = lambda h: ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem(RM))
a, b, p = put(a_h), put(b_h), put(p_h)
CASES = [
    ("add, ROW_MAJOR output, control", lambda o: ttnn.add(a, b, memory_config=mem(o)), RM, a_h + b_h),
    ("add, COL_MAJOR output", lambda o: ttnn.add(a, b, memory_config=mem(o)), CM, a_h + b_h),
    ("where, ROW_MAJOR output, control", lambda o: ttnn.where(p, a, b, memory_config=mem(o)), RM,
     torch.where(p_h != 0, a_h, b_h)),
    ("where, COL_MAJOR output", lambda o: ttnn.where(p, a, b, memory_config=mem(o)), CM,
     torch.where(p_h != 0, a_h, b_h)),
]
for name, op, orientation, want in CASES:
    try:
        wrong = int((ttnn.to_torch(op(orientation)) != want).sum())
        print(f"{name}: {wrong} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        print(f"{name}: {next((l for l in lines if l.startswith('Kernel')), lines[0])[:170]}", flush=True)
ttnn.close_device(device)
