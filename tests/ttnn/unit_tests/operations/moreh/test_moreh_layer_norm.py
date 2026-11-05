# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tt_py_test_utils_common.utility_functions import comp_allclose, skip_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_layer_norm import (
    make_input_tensors,
    run_moreh_layer_norm,
    run_moreh_layer_norm_backward,
    run_moreh_layer_norm_backward_with_gamma_or_beta,
    torch_layer_norm,
    torch_layer_norm_backward,
)
from tests.ttnn.unit_tests.operations.test_utils import (
    create_ttnn_tilized_tensor,
    get_compute_kernel_options,
    to_torch,
    to_ttnn,
)

pytestmark = pytest.mark.use_module_device

# With normalized_dims=1 the op writes mean/rstd wrong when they hold more than one value, so only single-row inputs
# request them.


# Not the nightly helpers: their zero padding hides a broken mask; NaN padding makes it fail.
def run_moreh_layer_norm_nan_pad_test(input_shape, normalized_dims, device):
    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, normalized_dims, True)
    expected_output, expected_mean, expected_rstd = torch_layer_norm(
        cpu_input, normalized_dims=normalized_dims, eps=1e-5, gamma=cpu_gamma, beta=cpu_beta
    )

    mean_rstd_shape = input_shape[:-normalized_dims]
    # NaN, so an unwritten value fails.
    tt_output = to_ttnn(torch.full(input_shape, float("nan")), device=device)
    tt_mean = to_ttnn(torch.full(mean_rstd_shape, float("nan")), device=device)
    tt_rstd = to_ttnn(torch.full(mean_rstd_shape, float("nan")), device=device)
    ttnn.moreh_layer_norm(
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        normalized_dims,
        1e-5,
        to_ttnn(cpu_gamma, device=device),
        to_ttnn(cpu_beta, device=device),
        output=tt_output,
        mean=tt_mean,
        rstd=tt_rstd,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    tol = 0.1 if normalized_dims == 1 else 0.15
    passing, out = comp_allclose(expected_output, to_torch(tt_output, shape=input_shape), rtol=tol, atol=tol)
    assert passing, out
    for expected, actual in [(expected_mean, tt_mean), (expected_rstd, tt_rstd)]:
        passing, out = comp_allclose(expected, to_torch(actual, shape=mean_rstd_shape), rtol=0.1, atol=0.1)
        assert passing, out


def run_moreh_layer_norm_backward_nan_pad_test(input_shape, normalized_dims, device):
    cpu_input, cpu_gamma, cpu_beta, cpu_output_grad = make_input_tensors(
        input_shape, normalized_dims, True, do_backward=True
    )
    expected_input_grad, expected_gamma_grad, expected_beta_grad = torch_layer_norm_backward(
        cpu_input.clone(),
        cpu_output_grad,
        normalized_dims=normalized_dims,
        eps=1e-5,
        gamma=cpu_gamma.clone(),
        beta=cpu_beta.clone(),
    )
    mean_rstd_dims = list(range(-normalized_dims, 0))
    mean = cpu_input.mean(dim=mean_rstd_dims, keepdim=True)
    rstd = (((cpu_input - mean) ** 2).mean(dim=mean_rstd_dims, keepdim=True) + 1e-5).rsqrt()

    mean_rstd_shape = input_shape[:-normalized_dims]
    normalized_shape = input_shape[-normalized_dims:]
    # NaN, so an unwritten value fails.
    tt_input_grad = to_ttnn(torch.full(input_shape, float("nan")), device=device)
    tt_gamma_grad = to_ttnn(torch.full(normalized_shape, float("nan")), device=device)
    tt_beta_grad = to_ttnn(torch.full(normalized_shape, float("nan")), device=device)
    ttnn.moreh_layer_norm_backward(
        create_ttnn_tilized_tensor(cpu_output_grad, device, ttnn.bfloat16),
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        to_ttnn(mean, device=device, shape=mean_rstd_shape),
        to_ttnn(rstd, device=device, shape=mean_rstd_shape),
        normalized_dims,
        gamma=to_ttnn(cpu_gamma, device=device),
        input_grad=tt_input_grad,
        gamma_grad=tt_gamma_grad,
        beta_grad=tt_beta_grad,
        compute_kernel_config=get_compute_kernel_options(False),
    )

    passing, out = comp_allclose(expected_input_grad, to_torch(tt_input_grad, shape=input_shape), rtol=0.1, atol=0.5)
    assert passing, out
    for expected, actual in [(expected_gamma_grad, tt_gamma_grad), (expected_beta_grad, tt_beta_grad)]:
        passing, out = comp_allclose(expected, to_torch(actual, shape=normalized_shape), rtol=0.1, atol=0.5)
        assert passing, out


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims",
    [
        ([1, 32], 1),
        ([2, 64, 64], 2),
    ],
    ids=["lastdim_aligned", "hw_aligned"],
)
def test_moreh_layer_norm(input_shape_normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(input_shape_normalized_dims, False, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims",
    [
        ([1, 20], 1),
        ([2, 77, 109], 2),
    ],
    ids=["lastdim", "hw"],
)
def test_moreh_layer_norm_unaligned(input_shape, normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_nan_pad_test(input_shape, normalized_dims, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, fp32_dest_acc_en",
    [
        (([2, 3, 45, 45], 3), False),
        (([1, 20], 1), True),
        (([1, 2, 500, 1000], 2), False),
    ],
    ids=["normalized_dims_3", "fp32_dest_acc", "large_algorithm"],
)
def test_moreh_layer_norm_corner_cases(input_shape_normalized_dims, fp32_dest_acc_en, device):
    torch.manual_seed(0)
    run_moreh_layer_norm(
        input_shape_normalized_dims, True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=fp32_dest_acc_en
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_allocated_output_no_mean_rstd(device):
    torch.manual_seed(0)
    # Not run_moreh_layer_norm: it always passes `output` and doesn't check that mean/rstd come back as None.
    input_shape = [3, 77, 109]
    cpu_input, cpu_gamma, cpu_beta, _ = make_input_tensors(input_shape, 1, True)
    expected_output, _, _ = torch_layer_norm(cpu_input, normalized_dims=1, eps=1e-5, gamma=cpu_gamma, beta=cpu_beta)
    tt_output, tt_mean, tt_rstd = ttnn.moreh_layer_norm(
        create_ttnn_tilized_tensor(cpu_input, device, ttnn.bfloat16),
        1,
        1e-5,
        to_ttnn(cpu_gamma, device=device),
        to_ttnn(cpu_beta, device=device),
        compute_kernel_config=get_compute_kernel_options(False),
    )

    assert tt_mean is None and tt_rstd is None
    passing, out = comp_allclose(expected_output, to_torch(tt_output, shape=input_shape), rtol=0.1, atol=0.1)
    assert passing, out


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape_normalized_dims, elementwise_affine",
    [
        (([6, 64, 64], 2), True),
        (([20, 30], 1), False),
    ],
    ids=["hw_aligned_batch", "no_affine"],
)
def test_moreh_layer_norm_backward(input_shape_normalized_dims, elementwise_affine, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(
        input_shape_normalized_dims, elementwise_affine, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "input_shape, normalized_dims",
    [
        ([20, 30], 1),
        ([20, 30], 2),
    ],
    ids=["lastdim", "hw"],
)
def test_moreh_layer_norm_backward_unaligned(input_shape, normalized_dims, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_nan_pad_test(input_shape, normalized_dims, device)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
@pytest.mark.parametrize("gamma_or_beta", [True, False], ids=["gamma_only", "beta_only"])
def test_moreh_layer_norm_backward_gamma_or_beta(gamma_or_beta, device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward_with_gamma_or_beta(
        ([20, 30], 1), gamma_or_beta, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_fp32_dest_acc(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(([20, 30], 1), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=True)


@skip_for_blackhole("Mismatching on BH, see #12349")
@pytest.mark.merge_gate
def test_moreh_layer_norm_backward_large_algorithm(device):
    torch.manual_seed(0)
    run_moreh_layer_norm_backward(
        ([1, 2, 500, 1000], 2), True, 1e-5, ttnn.bfloat16, device, compute_kernel_options=False
    )


@pytest.mark.merge_gate
def test_moreh_layer_norm_fp32_near_constant(device):
    import zlib

    import numpy as np

    def sample(name):
        seed = np.random.SeedSequence([20261105, zlib.crc32(name.encode())])
        return torch.from_numpy(np.random.Generator(np.random.PCG64(seed)).standard_normal(8).astype("float32"))

    x = (1 + 0.003 * sample("x")).to(torch.bfloat16).reshape(1, 1, 1, 8)
    gamma = (1 + 0.1 * sample("gamma").to(torch.bfloat16).float()).to(torch.bfloat16)
    beta = sample("beta").to(torch.bfloat16)
    expected = torch.nn.functional.layer_norm(x.double(), (8,), gamma.double(), beta.double(), 1e-5)

    def to_device(tensor):
        return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.moreh_layer_norm(
        to_device(x),
        1,
        1e-5,
        to_device(gamma),
        to_device(beta),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )[0]
    actual = ttnn.to_torch(actual).reshape(x.shape).double()
    relative_rms = (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()
    assert relative_rms < 0.01


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "shape, normalized_dims",
    [
        ((1, 1, 32 * 256, 64), 1),
        ((256, 1, 32, 8), 2),
    ],
    ids=["rows-and-width-tiles", "masked-scalar"],
)
def test_moreh_layer_norm_fp32_near_constant_reused_anchor(shape, normalized_dims, device):
    import zlib

    import numpy as np

    seed = np.random.SeedSequence([20261105, zlib.crc32(b"x")])
    sample = torch.from_numpy(np.random.Generator(np.random.PCG64(seed)).standard_normal(8).astype("float32"))
    values = (1 + 0.003 * sample).to(torch.bfloat16)
    x = values.repeat(torch.tensor(shape).prod().item() // 8).reshape(shape).float()
    outer_shape = shape[:-normalized_dims]
    offsets = (torch.arange(torch.tensor(outer_shape).prod().item()) % 2).reshape(
        *outer_shape, *((1,) * normalized_dims)
    )
    x = (x - offsets * 0.125).to(torch.bfloat16)
    # More outer tiles than cores forces at least one core to reuse the anchor buffer.
    expected = torch.nn.functional.layer_norm(x.double(), shape[-normalized_dims:], eps=1e-5)
    input_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.moreh_layer_norm(
        input_tensor,
        normalized_dims,
        1e-5,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )[0]
    actual = ttnn.to_torch(output).reshape(shape).double()
    relative_rms = (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()
    assert relative_rms < 0.01


@pytest.mark.merge_gate
def test_moreh_layer_norm_fp32_masked_alternating_offsets(device):
    shape = (256, 1, 17, 31)
    indices = torch.arange(256 * 17 * 31).reshape(shape)
    values = torch.tensor([1.0, 1.0, 0.99609375, 1.0, 1.0, 1.0, 1.0, 1.0])
    x = (values[indices % 8] - ((indices // (17 * 31)) % 2) * 0.125).to(torch.bfloat16)
    expected = torch.nn.functional.layer_norm(x.double(), shape[-2:], eps=1e-5)
    input_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.moreh_layer_norm(
        input_tensor,
        2,
        1e-5,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )[0]
    actual = ttnn.to_torch(output).reshape(shape).double()
    relative_rms = (torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)).item()
    assert relative_rms < 0.014142135623730172


@pytest.mark.merge_gate
@pytest.mark.parametrize("value", [2.0**127, 1.0 + 2.0**-4], ids=["near-float-max", "near-constant"])
def test_moreh_layer_norm_fp32_constant_row_mean(value, device):
    # A constant row of 64 values (two width tiles) with FP32 accumulation and a requested mean: the mean must be the value.
    # 2**127 is the largest power of two the sum of two tiles can hold only if it is scaled first.
    x = torch.full((1, 1, 1, 64), value).to(torch.bfloat16)
    tt_mean = to_ttnn(torch.full((1, 1, 1), float("nan")), device=device)
    ttnn.moreh_layer_norm(
        to_ttnn(x, device=device),
        1,
        1e-5,
        mean=tt_mean,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        ),
    )
    actual = to_torch(tt_mean, shape=(1, 1, 1)).double()
    assert torch.isfinite(actual).all()
    assert abs(actual.item() - value) <= 0.01 * value
