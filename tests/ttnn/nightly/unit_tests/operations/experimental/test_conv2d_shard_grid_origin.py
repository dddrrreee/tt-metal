# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


def width_sharded(start, size, shard):
    # ttnn.conv2d reshards a width-sharded input that is not ROW_MAJOR onto (0, 0), so only ROW_MAJOR grids reach it.
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(start[0] + size[0] - 1, start[1] + size[1] - 1))}
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )


def put(device, x, mem):
    return ttnn.from_torch(x, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, memory_config=mem)


def values(shape, op, start, salt=0):
    # Distinct placement data prevents stale L1 contents from hiding misplaced reads.
    seed = 100000 * start[0] + 10000 * start[1] + 100 * salt + sum((i + 1) * ord(c) for i, c in enumerate(op))
    return torch.randint(-1, 2, shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def conv2d_case(device, op, start, expect_error):
    if op.endswith("2x2"):
        # Four width shards on 2x2 cores: the factory walks across the device width, so this grid is rejected.
        x = values((1, 128, 8, 8), op, start)
        a = put(device, x.permute(0, 2, 3, 1).reshape(1, 1, 64, 128), width_sharded(start, (2, 2), (64, 32)))
        with expect_error(RuntimeError, "row by row across the device grid"):
            ttnn.conv2d(
                input_tensor=a,
                weight_tensor=ttnn.from_torch(values((64, 128, 3, 3), op, start, salt=1), dtype=ttnn.bfloat16),
                in_channels=128,
                out_channels=64,
                device=device,
                kernel_size=(3, 3),
                batch_size=1,
                input_height=8,
                input_width=8,
                conv_config=ttnn.Conv2dConfig(
                    weights_dtype=ttnn.bfloat16,
                    shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    reshard_if_not_optimal=False,
                ),
            )
        return
    x = values((1, 64, 8, 8), op, start)
    w = values((64, 64, 3, 3), op, start, salt=1)
    # Unpadded row-major width-sharded input skips halo; the sums stay small integers, exact in bfloat16.
    ref = torch.nn.functional.conv2d(x.float(), w.float()).permute(0, 2, 3, 1).reshape(1, 1, 36, 64)
    a = put(device, x.permute(0, 2, 3, 1).reshape(1, 1, 64, 64), width_sharded(start, (2, 1), (64, 32)))

    def run():
        return ttnn.conv2d(
            input_tensor=a,
            weight_tensor=ttnn.from_torch(w, dtype=ttnn.bfloat16),
            in_channels=64,
            out_channels=64,
            device=device,
            kernel_size=(3, 3),
            batch_size=1,
            input_height=8,
            input_width=8,
            conv_config=ttnn.Conv2dConfig(
                weights_dtype=ttnn.bfloat16, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
            ),
            compute_config=ttnn.init_device_compute_kernel_config(
                device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
            ),
        )

    if start != (0, 0):
        with expect_error(RuntimeError, "row by row across the device grid"):
            run()
        return
    got = ttnn.to_torch(run()).float()
    assert got.shape == ref.shape, (got.shape, ref.shape)
    wrong = got != ref
    assert not wrong.any(), dict(wrong=int(wrong.sum()), elements=got.numel(), max_abs=float((got - ref).abs().max()))


FOUR_ORIGINS = [(0, 0), (1, 0), (0, 1), (2, 3)]
CASES = [("conv2d_width", origin) for origin in FOUR_ORIGINS] + [("conv2d_width_2x2", (0, 0))]


@pytest.mark.parametrize("op,origin", CASES)
def test_conv2d_shard_grid_origin(device, op, origin, expect_error):
    conv2d_case(device, op, origin, expect_error)
