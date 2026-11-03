"""ttnn.conv2d with a pre-sharded WIDTH_SHARDED input whose shard grid is not one row starting at core (0, 0).

The width-sharded conv factory places core i's runtime args on (i % grid_x, i / grid_x), where grid_x is the
device's compute-grid width (conv2d_op_width_sharded_program_factory.cpp), and its reader rebuilds the same
index (activation_reader_width_sharded.cpp). Buffers place channel slice i on
corerange_to_cores(grid, n, row_major)[i]. A pre-sharded input keeps its own grid. One Blackhole chip or ttsim.
The weights copy channel o from the centre tap of a 3x3 kernel, so the output is the input's 14x14 interior,
exact in bfloat16 with HiFi4.
usage: python repro_conv2d_width_sharded.py [CASE_INDEX ...]   (default: all; a case that does not complete hides
the later ones, so run those one per process)."""
import sys

import torch
import ttnn

CC = ttnn.CoreCoord


def grid(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(CC(*a), CC(*b)) for a, b in ranges})


CASES = [  # (name, input shard grid, out_channels); input NHWC [1, 16, 16, 256], 8 channel slices of 32
    ("8 cores (0, 0)-(7, 0), control", grid(((0, 0), (7, 0))), 256),
    ("8 cores (1, 0)-(8, 0)", grid(((1, 0), (8, 0))), 256),
    ("4x2 cores (0, 0)-(3, 1)", grid(((0, 0), (3, 1))), 256),
    ("4x2 cores (0, 0)-(3, 1), 128 output channels", grid(((0, 0), (3, 1))), 128),
]

device = ttnn.open_device(device_id=0, l1_small_size=32768)
compute = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=False)
SELECTED = [CASES[int(i)] for i in sys.argv[1:]] or CASES
for name, shard_grid, out_c in SELECTED:
    torch.manual_seed(0)
    x = (torch.randint(-16, 17, (1, 16, 16, 256)) / 8).to(torch.bfloat16)
    w = torch.zeros(out_c, 256, 3, 3, dtype=torch.bfloat16)
    w[torch.arange(out_c), torch.arange(out_c), 1, 1] = 1  # output channel o = input channel o at the centre tap
    want = x[:, 1:15, 1:15, :out_c].float()
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(shard_grid, [256, 32], ttnn.ShardOrientation.ROW_MAJOR))
    tx = ttnn.from_torch(x.reshape(1, 1, 256, 256), layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16)
    print(f"conv2d, {name}: calling", flush=True)
    try:
        y = ttnn.conv2d(input_tensor=tx, weight_tensor=tw, device=device, in_channels=256, out_channels=out_c,
                        batch_size=1, input_height=16, input_width=16, kernel_size=(3, 3), stride=(1, 1),
                        padding=(0, 0), conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16),
                        compute_config=compute)
        got = ttnn.to_torch(y).float().reshape(-1, out_c)[: 14 * 14].reshape(want.shape)
        print(f"conv2d, {name}: {int((got != want).sum())} of {want.numel()} values wrong", flush=True)
    except RuntimeError as e:
        lines = [l.strip() for l in str(e).splitlines() if l.strip()]
        msg = next((l for l in lines if l.startswith('Kernel')), None) or next(
            (l for l in lines if not l.startswith(('TT_FATAL @', 'TT_THROW @', 'info:', 'backtrace'))), lines[0])
        print(f"conv2d, {name}: {msg[:170]}", flush=True)
ttnn.close_device(device)
