# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import comp_allclose_and_pcc, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.reduction.utility_functions import ttnn_prod

# Module-scoped device: these tests all run with the default device config, so the device is
# opened once per file instead of once per test case.
pytestmark = pytest.mark.use_module_device

TILE_HEIGHT = 32
TILE_WIDTH = 32


def get_tensors(input_shape, output_shape, device, npu_dtype=ttnn.bfloat16):
    torch.manual_seed(2023)
    cpu_dtype = torch.float32 if npu_dtype == ttnn.float32 else torch.bfloat16
    npu_layout = ttnn.TILE_LAYOUT

    torch_input = torch.randint(-100, 100, input_shape, dtype=cpu_dtype)
    torch_output = torch.randint(-100, 100, output_shape, dtype=cpu_dtype)

    tt_input = ttnn.Tensor(torch_input, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)
    tt_output = ttnn.Tensor(torch_output, npu_dtype).pad_to_tile(float("nan")).to(npu_layout).to(device)

    return tt_input, tt_output, torch_input


@pytest.mark.parametrize(
    "input_shape",
    (
        ([2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1]),
        ([9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1]),
        ([4, 3, TILE_HEIGHT * 3 - 1, TILE_WIDTH * 11 - 1]),
        ([1, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1]),
        ([4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1]),
        ([4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1]),
        ([8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1]),
    ),
    ids=[
        "2, 3, TILE_HEIGHT * 6 - 1, TILE_WIDTH * 7 - 1",
        "9, 16, TILE_HEIGHT * 13 - 1, TILE_WIDTH * 19 - 1",
        "4, 3, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 11 - 1",
        "1, 1, TILE_HEIGHT-1,TILE_WIDTH - 1",
        "4, 4, TILE_HEIGHT * 9 - 1, TILE_WIDTH * 12 - 1",
        "4, 4, TILE_HEIGHT * 12 - 1, TILE_WIDTH * 9 - 1",
        "8, 8, TILE_HEIGHT * 4 - 1, TILE_WIDTH * 4 - 1",
    ],
)
@pytest.mark.parametrize(
    "dims",
    (
        [
            0,
        ],
        [
            1,
        ],
    ),
    ids=["0", "1"],
)
@pytest.mark.parametrize("npu_dtype", (ttnn.bfloat16, ttnn.float32), ids=["bfloat16", "float32"])
# Support for dim 2,3 in composite_ops
def test_prod_dims(input_shape, dims, npu_dtype, device):
    output_shape = input_shape.copy()

    for dim in dims:
        output_shape[dim] = 1

    (tt_input, tt_output, torch_input) = get_tensors(input_shape, output_shape, device, npu_dtype)

    torch_output = torch.prod(torch_input, dims[0], True)

    cpu_layout = ttnn.ROW_MAJOR_LAYOUT
    tt_output_cpu = ttnn_prod(tt_input, tt_output, dims=dims).cpu().to(cpu_layout).to_torch()

    rtol = atol = 0.1
    passing, output_pcc = comp_allclose_and_pcc(torch_output, tt_output_cpu, pcc=0.999, rtol=rtol, atol=atol)

    logger.info(f"Out passing={passing}")
    logger.info(f"Output pcc={output_pcc}")

    assert passing


DIM_PATH_BLOCK_FLOAT_SUPPORT = {
    ttnn.bfloat8_b: True,
    ttnn.bfloat4_b: False,
}
_dim_supported = [d for d, ok in DIM_PATH_BLOCK_FLOAT_SUPPORT.items() if ok]
_dim_unsupported = [d for d, ok in DIM_PATH_BLOCK_FLOAT_SUPPORT.items() if not ok]


@pytest.mark.parametrize("npu_dtype", _dim_supported, ids=lambda d: d.name.lower())
@pytest.mark.parametrize("dim", (0, 1, 2, 3))
def test_prod_dims_block_float(dim, npu_dtype, device):
    # Mostly-ones input with a few 2.0s: exact in block-float, product stays a small power of two.
    torch.manual_seed(2023)
    torch_input = torch.ones([2, 3, TILE_HEIGHT, TILE_WIDTH * 2], dtype=torch.float32)
    torch_input.view(-1)[::13][:8] = 2.0

    tt_input = ttnn.from_torch(torch_input, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.prod(tt_input, dim=dim)
    assert tt_output.dtype == npu_dtype, f"expected {npu_dtype} result, got {tt_output.dtype}"

    out = ttnn.to_torch(tt_output)
    ref = torch.prod(torch_input, dim=dim)
    ref = ttnn.to_torch(ttnn.from_torch(ref, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device))
    assert torch.allclose(out, ref, atol=1e-2), f"dim={dim} {npu_dtype}: expected {ref}, got {out}"


@pytest.mark.parametrize(
    "input_shape, output_shape, dims",
    (
        ([1, 0, TILE_HEIGHT, TILE_WIDTH], [1, 1, TILE_HEIGHT, TILE_WIDTH], [1]),
        ([0, 2, TILE_HEIGHT, TILE_WIDTH], [1, 2, TILE_HEIGHT, TILE_WIDTH], [0]),
        ([2, 0, TILE_HEIGHT - 1, TILE_WIDTH - 1], [2, 1, TILE_HEIGHT - 1, TILE_WIDTH - 1], [1]),
        ([2, 0, TILE_HEIGHT, TILE_WIDTH], [1, 1, TILE_HEIGHT, TILE_WIDTH], [0, 1]),
        ([0, 2, TILE_HEIGHT, TILE_WIDTH], [0, 1, TILE_HEIGHT, TILE_WIDTH], [1]),
    ),
    ids=[
        "empty_dim1",
        "empty_dim0",
        "empty_dim1_unaligned_hw",
        "empty_multi_dim01",
        "empty_non_reduced_dim0",
    ],
)
@pytest.mark.parametrize(
    "npu_dtype", (ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b), ids=["bfloat16", "float32", "bfloat8_b"]
)
def test_prod_preallocated_zero_volume(input_shape, output_shape, dims, npu_dtype, device):
    # Without a zero-volume guard in prod_nc, reducing an empty axis launches the
    # compute kernel with num_input_tiles=0 and num_output_tiles>0, hanging in wait_front.
    torch_input = torch.empty(input_shape, dtype=torch.float32)
    torch_output_init = torch.full(output_shape, 7.0, dtype=torch.float32)
    expected = torch_input
    for dim in dims:
        expected = torch.prod(expected, dim=dim, keepdim=True)

    tt_input = ttnn.from_torch(torch_input, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.from_torch(torch_output_init, dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    returned = ttnn.prod(tt_input, tt_output, dims=dims)
    got_returned = ttnn.to_torch(returned).float()
    got_preallocated = ttnn.to_torch(tt_output).float()

    assert list(got_returned.shape) == output_shape
    assert list(got_preallocated.shape) == output_shape
    assert torch.equal(got_returned, expected)
    assert torch.equal(got_preallocated, expected)


def test_prod_preallocated_rejects_row_major_layout(device, expect_error):
    # ProdNcProgramFactory indexes tile-sized (2048 B for bf16) entries; a ROW_MAJOR
    # input or output would read or write past its row-major pages if not rejected.
    torch_input = torch.full((1, 2, TILE_HEIGHT, TILE_WIDTH), 2.0, dtype=torch.bfloat16)
    torch_output = torch.zeros((1, 1, TILE_HEIGHT, TILE_WIDTH), dtype=torch.bfloat16)

    tt_input_rm = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output_tile = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Input layout must be TILE"):
        ttnn.prod(tt_input_rm, tt_output_tile, dims=[1])

    tt_input_tile = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output_rm = ttnn.from_torch(torch_output, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "Output layout must be TILE"):
        ttnn.prod(tt_input_tile, tt_output_rm, dims=[1])


@pytest.mark.parametrize("npu_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("memory", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_prod_preallocated_empty_full_product(device, npu_dtype, memory):
    x = ttnn.from_torch(torch.empty([1, 0, 32, 32]), dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.from_torch(
        torch.tensor(7.0), dtype=npu_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory
    )
    result = ttnn.prod(x, output, dims=[])
    assert result.buffer_address() == output.buffer_address()
    assert tuple(result.shape) == ()
    assert ttnn.to_torch(result).item() == 1.0
    assert ttnn.to_torch(output).item() == 1.0


@pytest.mark.parametrize("input_shape", [[1, 0, 32, 32], [1, 2, 32, 32]], ids=["empty", "nonempty"])
@pytest.mark.parametrize("output_shape", [[1, 1, 31, 32], [1, 1, 32, 31], [1, 1, 31, 31]])
def test_prod_preallocated_rejects_incorrect_logical_shape(device, expect_error, input_shape, output_shape):
    x = ttnn.from_torch(torch.ones(input_shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.from_torch(torch.ones(output_shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "logical shapes must match"):
        ttnn.prod(x, output, dims=[1])


@pytest.mark.parametrize("dims", [[0, 1], [1, 0]])
@pytest.mark.parametrize("input_shape", [[2, 0, 31, 31], [2, 2, 31, 31]], ids=["empty", "nonempty"])
def test_prod_preallocated_multidim_unaligned(device, dims, input_shape):
    x = ttnn.from_torch(torch.ones(input_shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.from_torch(
        torch.full([1, 1, 31, 31], 7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    result = ttnn.prod(x, output, dims=dims)
    assert result.buffer_address() == output.buffer_address()
    assert torch.equal(ttnn.to_torch(output), torch.ones([1, 1, 31, 31], dtype=torch.bfloat16))


def test_prod_preallocated_empty_full_rejects_invalid_operands(device, expect_error):
    x = ttnn.from_torch(torch.empty([1, 0, 32, 32]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.from_torch(torch.tensor(7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    non_scalar = ttnn.from_torch(
        torch.ones([1, 1, 32, 32]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Full product output must be a scalar"):
        ttnn.prod(x, non_scalar, dims=[])
    rank_one = ttnn.from_torch(torch.tensor([7.0]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Full product output must be a scalar"):
        ttnn.prod(x, rank_one, dims=[])
    wrong_dtype = ttnn.from_torch(torch.tensor(7.0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "input and output dtypes to match"):
        ttnn.prod(x, wrong_dtype, dims=[])
    row_input = ttnn.from_torch(
        torch.empty([1, 0, 32, 32]), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Input layout must be TILE"):
        ttnn.prod(row_input, output, dims=[])
    row_output = ttnn.from_torch(torch.tensor(7.0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, "Output layout must be TILE"):
        ttnn.prod(x, row_output, dims=[])
    host_output = ttnn.from_torch(torch.tensor(7.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(RuntimeError, "Operands to prod_nc need to be on device"):
        ttnn.prod(x, host_output, dims=[])
    unsupported_input = ttnn.from_torch(
        torch.empty([1, 0, 32, 32], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    unsupported_output = ttnn.from_torch(
        torch.tensor(7, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "unsupported data type for prod"):
        ttnn.prod(unsupported_input, unsupported_output, dims=[])
    ttnn.deallocate(output)
    with expect_error(RuntimeError, "Operands to prod_nc need to be allocated"):
        ttnn.prod(x, output, dims=[])
