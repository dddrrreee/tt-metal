# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics


def run_moe_test(N, C, H, W, k, E, e, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input[:, :, :, E:] = 0  # padded input

    expert_mask = torch.zeros([N, C, 1, W], dtype=torch_dtype)
    expert_mask[:, :, :, E:] = float("-inf")

    topE_mask = torch.zeros([N, C, 1, k], dtype=torch_dtype)
    topE_mask[:, :, :, e:] = float("-inf")

    pyt_topk_values, pyt_topk_indices = torch.topk(input + expert_mask, k, dim=-1)
    torch_weights_1SB1 = torch.sum(
        (torch.softmax(pyt_topk_values + topE_mask, dim=-1) * (pyt_topk_indices == 0))[:, :, :, :e],
        dim=-1,
        keepdim=True,
    )
    ttnn_input = ttnn.from_torch(input, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_expert_mask = ttnn.from_torch(expert_mask, dtype, layout=ttnn.Layout.TILE, device=device)
    ttnn_topE_mask = ttnn.from_torch(topE_mask, dtype, layout=ttnn.Layout.TILE, device=device)

    for i in range(3):
        weights_1SB1 = ttnn.moe(ttnn_input, ttnn_expert_mask, ttnn_topE_mask, k)

        assert list(weights_1SB1.padded_shape) == [N, C, H, k]

        ttnn_weights_1SB1 = ttnn.to_torch(weights_1SB1)

        # test for equivalance
        assert_numeric_metrics(
            torch_weights_1SB1,
            ttnn_weights_1SB1,
            pcc_threshold=0.999,
            rtol=0.046,
            atol=0.012,
            frobenius_threshold=0.017,
            check_ulp=True,
            ulp_threshold=8,
        )


@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
    ids=[
        "BFLOAT16_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W, k, E, e",
    [
        (1, 1, 32, 64, 32, 8, 2),  # Mixtral8x7B decode shape (Wt=2)
        (1, 1, 32, 256, 32, 8, 2),  # wider expert dim (Wt=8) — exercises the upfront expert-mask load for Wt>4
    ],
)
def test_moe(N, C, H, W, k, E, e, dtype, device):
    run_moe_test(N, C, H, W, k, E, e, dtype, device)


def _moe_inputs(device):
    # A [1, 1, 64, 64] input: the result is [1, 1, 64, 1], two tile rows (padded shape [1, 1, 64, 32]).
    torch.manual_seed(0)
    topk_mask = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    topk_mask[..., 8:] = float("-inf")
    tensors = (
        torch.randn(1, 1, 64, 64, dtype=torch.bfloat16),
        torch.zeros(1, 1, 1, 64, dtype=torch.bfloat16),
        topk_mask,
    )
    return [ttnn.from_torch(t, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in tensors]


@pytest.mark.parametrize("out_shape", [[1, 1, 64, 1], [1, 1, 64, 32]])
def test_moe_output_tensor(out_shape, device):
    x, expert_mask, topk_mask = _moe_inputs(device)
    expected = ttnn.to_torch(ttnn.moe(x, expert_mask, topk_mask, 32))
    out = ttnn.from_torch(torch.zeros(out_shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.moe(x, expert_mask, topk_mask, 32, output_tensor=out)
    assert torch.equal(ttnn.to_torch(out)[..., :1], expected)


# One tile row is listed last: without the check, the program does not finish.
@pytest.mark.parametrize("out_shape", [[1, 1, 96, 32], [1, 1, 64, 64], [1, 2, 32, 32], [1, 1, 32, 32]])
def test_moe_rejects_output_tensor_of_another_shape(out_shape, device, expect_error):
    x, expert_mask, topk_mask = _moe_inputs(device)
    out = ttnn.from_torch(torch.zeros(out_shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must equal expected padded shape"):
        ttnn.moe(x, expert_mask, topk_mask, 32, output_tensor=out)


def test_moe_overpadded_input_output_tensor(device, expect_error):
    torch.manual_seed(0)
    x_host = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    topk_host = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
    topk_host[..., 8:] = float("-inf")
    x_unpadded = ttnn.from_torch(x_host, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expert_mask = ttnn.from_torch(
        torch.zeros(1, 1, 1, 64, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    topk_mask = ttnn.from_torch(topk_host, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    expected = ttnn.to_torch(ttnn.moe(x_unpadded, expert_mask, topk_mask, 32))

    x_rm = ttnn.from_torch(x_host, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    x_overpadded = ttnn.tilize_with_val_padding(x_rm, [1, 1, 64, 64], 0.0)

    out_logical_rm = ttnn.from_torch(
        torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out_logical = ttnn.tilize_with_val_padding(out_logical_rm, [1, 1, 64, 32], 0.0)
    ttnn.moe(x_overpadded, expert_mask, topk_mask, 32, output_tensor=out_logical)
    assert torch.equal(ttnn.to_torch(out_logical), expected)

    out_padded = ttnn.from_torch(
        torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn.moe(x_overpadded, expert_mask, topk_mask, 32, output_tensor=out_padded)
    assert torch.equal(ttnn.to_torch(out_padded)[..., :32, :1], expected)

    out_small = ttnn.from_torch(
        torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16), ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "must equal expected padded shape"):
        ttnn.moe(x_overpadded, expert_mask, topk_mask, 32, output_tensor=out_small)
