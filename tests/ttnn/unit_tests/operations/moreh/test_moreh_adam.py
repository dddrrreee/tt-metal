# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.moreh.test_moreh_adam import run_moreh_adam

pytestmark = pytest.mark.use_module_device


@pytest.mark.merge_gate
@pytest.mark.parametrize("shape", [[32, 32], [32, 149 * 32]], ids=["single_tile", "core_group_2"])
def test_moreh_adam(shape, device):
    torch.manual_seed(0)
    run_moreh_adam(shape, 0.1, (0.5, 0.555), 1e-8, 0.3, True, False, device)


@pytest.mark.merge_gate
@pytest.mark.parametrize(
    "lr, betas, weight_decay, amsgrad, fp32_dest_acc_en, step, param_atol",
    [
        (0.1, (0.5, 0.555), 0.3, False, False, 1, 0.01),
        (0.1, (0.5, 0.555), 0.3, True, True, 1, 0.01),
        (0.1, (0.5, 0.555), 0.3, False, True, 1, 0.01),
        # lr=1 so ignoring `step` misses by ~0.26; beta2=0.999 truncated to bfloat16 shifts the update by up to ~0.05.
        (1.0, (0.9, 0.999), 0.0, False, False, 2, 0.05),
        (1.0, (0.9, 0.999), 0.0, False, False, 10, 0.05),
    ],
    ids=["no_amsgrad", "fp32_dest_acc", "no_amsgrad_fp32_dest_acc", "step_2", "step_10"],
)
def test_moreh_adam_corner_cases(lr, betas, weight_decay, amsgrad, fp32_dest_acc_en, step, param_atol, device):
    torch.manual_seed(0)
    run_moreh_adam(
        [32, 32], lr, betas, 1e-8, weight_decay, amsgrad, fp32_dest_acc_en, device, step=step, param_atol=param_atol
    )


@pytest.mark.parametrize("output_name", ["param_out", "exp_avg_out", "exp_avg_sq_out", "max_exp_avg_sq_out"])
@pytest.mark.parametrize("output_shape", [[32, 64], [31, 32], [1, 32, 32]])
def test_moreh_adam_rejects_wrong_output_shape(device, expect_error, output_name, output_shape):
    def tensor(shape):
        return ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    inputs = [tensor([32, 32]) for _ in range(5)]
    with expect_error(RuntimeError, f"{output_name} must have the same logical shape as param_in"):
        ttnn.moreh_adam(
            *inputs[:4], step=1, amsgrad=True, max_exp_avg_sq_in=inputs[4], **{output_name: tensor(output_shape)}
        )


def test_moreh_adam_amsgrad_requires_max_input(device, expect_error):
    tensor = ttnn.from_torch(torch.zeros([32, 32], dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "amsgrad=True requires max_exp_avg_sq_in"):
        ttnn.moreh_adam(tensor, tensor, tensor, tensor, step=1, amsgrad=True)
