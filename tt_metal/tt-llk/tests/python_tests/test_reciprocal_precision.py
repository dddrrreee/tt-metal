# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hold precise FP32 reciprocal to one step, including the underflowing correction."""

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.test_config import TestConfig
from helpers.ulp import ulp_distance
from test_eltwise_unary_sfpu import build_eltwise_unary_sfpu_config


@pytest.mark.parametrize("input_format", [DataFormat.Float32, DataFormat.Float16_b])
@pytest.mark.skipif(
    TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE,
    reason="Blackhole full-tile reciprocal regression",
)
def test_reciprocal_precise_binades(input_format):
    # A different sample from the budget emitter: random mantissas plus exact
    # powers of two and adjacent values in every normal binade, for both signs.
    generator = torch.Generator().manual_seed(62655)
    mantissas = torch.randint(0, 1 << 23, (254, 128), generator=generator)
    mantissas[:, :4] = torch.tensor([0, 1, (1 << 23) - 2, (1 << 23) - 1])
    if input_format == DataFormat.Float16_b:
        mantissas = (mantissas >> 16) << 16
    bits = (torch.arange(1, 255)[:, None] << 23) | mantissas
    positive = bits.to(torch.int32).flatten().view(torch.float32)
    src = torch.cat([positive, -positive, torch.ones(512)])
    if input_format == DataFormat.Float16_b:
        src = src.to(torch.bfloat16)
    formats = InputOutputFormat(input_format, DataFormat.Float32)
    config = build_eltwise_unary_sfpu_config(
        formats,
        DestAccumulation.Yes,
        ApproximationMode.No,
        MathOperation.Reciprocal,
        [32, 2048],
        src,
        torch.zeros_like(src),
        64,
        64,
    )
    result = torch.tensor(config.run().result, dtype=torch.float32)
    assert torch.isfinite(result).all(), "normal inputs must give finite reciprocals"
    golden = src.double().reciprocal().float()
    steps = ulp_distance(result, golden, flush_subnormals=True)
    worst = int(steps.max())
    assert worst <= 1, f"precise reciprocal: {worst} steps; expected at most one"
