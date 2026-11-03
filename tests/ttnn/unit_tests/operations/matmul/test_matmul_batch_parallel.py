# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@pytest.mark.parametrize(
    "batch_shape,m,k,n",
    [
        ((32,), 32, 64, 32),
        ((4,), 64, 64, 64),
        ((64,), 64, 512, 64),
        ((2, 16), 128, 128, 128),
        ((17,), 160, 288, 96),
        ((8,), 33, 65, 47),
        ((2, 2, 2), 63, 111, 65),
        ((9,), 96, 96, 96),
        ((4,), 64, 704, 64),
        ((32,), 32, 128, 1024),
    ],
)
def test_matmul_default_batch_outputs_and_cache(device, isolate_program_cache, batch_shape, m, k, n):
    for seed in (0, 1):
        torch.manual_seed(seed)
        a = torch.randn((*batch_shape, m, k), dtype=torch.bfloat16)
        b = torch.randn((*batch_shape, k, n), dtype=torch.bfloat16)
        ta = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tb = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        actual = ttnn.to_torch(ttnn.matmul(ta, tb))
        assert actual.shape == a.shape[:-1] + (n,)
        assert_with_pcc(a.float() @ b.float(), actual.float(), 0.999)
    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize(
    "dtype,memory_config",
    [
        (ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
        (ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG),
        (ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG),
        (ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_matmul_small_batches_other_formats(device, dtype, memory_config):
    torch.manual_seed(0)
    a = torch.randn((4, 64, 64))
    b = torch.randn((4, 64, 64))
    ta = ttnn.from_torch(a, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    tb = ttnn.from_torch(b, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    actual = ttnn.to_torch(ttnn.matmul(ta, tb, memory_config=memory_config))
    assert_with_pcc(a @ b, actual.float(), 0.99)


@pytest.mark.parametrize("bias_shape", [(1, 1, 1, 64), (1, 1, 64, 64)])
def test_linear_default_batch_outputs_with_bias(device, bias_shape):
    torch.manual_seed(0)
    a = torch.randn((8, 1, 64, 64), dtype=torch.bfloat16)
    b = torch.randn((8, 1, 64, 64), dtype=torch.bfloat16)
    bias = torch.randn(bias_shape, dtype=torch.bfloat16)
    ta, tb, tt_bias = [
        ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in (a, b, bias)
    ]
    actual = ttnn.to_torch(ttnn.linear(ta, tb, bias=tt_bias))
    assert_with_pcc(a.float() @ b.float() + bias.float(), actual.float(), 0.999)


@pytest.mark.parametrize("n", [32, 1056, 2048, 3488])
def test_matmul_default_wide_and_large_k(device, isolate_program_cache, n):
    torch.manual_seed(63291)
    batch, k = (4, 4096) if n == 32 else (n // 32, 128)
    a = torch.randn((batch, 32, k), dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), dtype=torch.bfloat16)
    ta, tb = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in (a, b)]
    actual = ttnn.to_torch(ttnn.matmul(ta, tb))
    assert_with_pcc(a.float() @ b.float(), actual.float(), 0.999)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("placement", ["offset", "worker_rows", "implicit_id"])
def test_matmul_default_with_subdevice(device, isolate_program_cache, placement):
    grid = device.compute_with_storage_grid_size()
    if placement == "implicit_id":
        worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
        subdevices = [ttnn.SubDevice([worker])]
        kwargs = {}
    elif placement == "offset":
        worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3))})
        subdevices = [ttnn.SubDevice([worker])]
        kwargs = {"sub_device_id": ttnn.SubDeviceId(0)}
    else:
        dummy = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 0))})
        worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        subdevices = [ttnn.SubDevice([dummy]), ttnn.SubDevice([worker])]
        kwargs = {"sub_device_id": ttnn.SubDeviceId(1)}
    manager = device.create_sub_device_manager(subdevices, 0)
    device.load_sub_device_manager(manager)
    device.set_sub_device_stall_group([ttnn.SubDeviceId(i) for i in range(len(subdevices))])
    try:
        torch.manual_seed(63291)
        a, b = [torch.randn((4, 64, 64), dtype=torch.bfloat16) for _ in range(2)]
        ta, tb = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in (a, b)]
        actual = ttnn.to_torch(ttnn.matmul(ta, tb, **kwargs))
        assert_with_pcc(a.float() @ b.float(), actual.float(), 0.999)
    finally:
        device.disable_and_clear_program_cache()
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(manager)


@pytest.mark.parametrize("transpose_tile_operand", ["a", "b"])
def test_matmul_default_with_transposed_tile(device, transpose_tile_operand):
    torch.manual_seed(63291)
    k = device.compute_with_storage_grid_size().x * 60 * 32
    a = torch.randn((8, 96, k), dtype=torch.bfloat16)
    b = torch.randn((8, k, 64), dtype=torch.bfloat16)
    ta, tb = [
        ttnn.from_torch(
            t,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            tile=ttnn.Tile((32, 32), transpose_tile=name == transpose_tile_operand),
        )
        for name, t in (("a", a), ("b", b))
    ]
    actual = ttnn.to_torch(ttnn.matmul(ta, tb))
    assert_with_pcc(a.float() @ b.float(), actual.float(), 0.999)


def test_matmul_default_circular_buffer_fallback(device):
    torch.manual_seed(63291)
    k = device.compute_with_storage_grid_size().x * 60 * 32
    a = torch.randn((16, 128, k), dtype=torch.bfloat16)
    b = torch.randn((16, k, 128), dtype=torch.bfloat16)
    ta, tb = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in (a, b)]
    actual = ttnn.to_torch(ttnn.matmul(ta, tb))
    assert_with_pcc(a.float() @ b.float(), actual.float(), 0.999)
