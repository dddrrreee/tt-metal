import argparse
import torch
import ttnn

p = argparse.ArgumentParser()
p.add_argument('--explicit-id', action='store_true')
args = p.parse_args()
d = ttnn.open_device(device_id=0)
cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
manager = d.create_sub_device_manager([ttnn.SubDevice([cores])], 0)
d.load_sub_device_manager(manager)
d.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
try:
    torch.manual_seed(63291)
    a, b = [torch.randn((4, 64, 64), dtype=torch.bfloat16) for _ in range(2)]
    ta, tb = [ttnn.from_torch(x, device=d, layout=ttnn.TILE_LAYOUT) for x in (a, b)]
    kw = {'sub_device_id': ttnn.SubDeviceId(0)} if args.explicit_id else {}
    actual = ttnn.to_torch(ttnn.matmul(ta, tb, **kw)).float()
    expected = a.float() @ b.float()
    print('PASS PCC', torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item(), flush=True)
finally:
    d.disable_and_clear_program_cache()
    d.reset_sub_device_stall_group()
    d.clear_loaded_sub_device_manager()
    d.remove_sub_device_manager(manager)
    ttnn.close_device(d)
