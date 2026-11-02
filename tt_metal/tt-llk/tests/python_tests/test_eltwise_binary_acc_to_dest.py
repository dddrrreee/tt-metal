# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multiply into a live destination, so overwrite cannot rely on acquire clearing it."""

from copy import deepcopy

import pytest
import torch
from fuser.config_parser import FuserConfigSchema
from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.test_config import TestConfig

pytestmark = pytest.mark.skipif(
    TestConfig.CHIP_ARCH != ChipArchitecture.BLACKHOLE,
    reason="Blackhole ELWMUL destination accumulation contract",
)


@pytest.mark.parametrize("fidelity", list(MathFidelity))
@pytest.mark.parametrize("acc_to_dest", [False, True])
@pytest.mark.parametrize(
    "data_format,dest_acc",
    [
        ("Float16_b", DestAccumulation.No),
        ("Float16_b", DestAccumulation.Yes),
        ("Float32", DestAccumulation.Yes),
    ],
)
@pytest.mark.parametrize("tile_rows", [8, 32])
@pytest.mark.parametrize("dest_sync", ["SyncFull", "SyncHalf"])
def test_elwmul_acc_to_dest(
    fidelity, acc_to_dest, data_format, dest_acc, tile_rows, dest_sync
):
    definition = FuserConfigSchema.load_definition("hifi_elwmul_overwrite")
    definition["dest_acc"] = dest_acc
    definition["loop_factor"] = 1
    definition["operations"][0]["dest_sync"] = dest_sync
    # Fill both DEST halves, including the last slot, with a nonzero seed.
    tiles = 8 if dest_acc == DestAccumulation.Yes else 16
    for operand in definition["operands"]:
        operand["dims"] = [tile_rows, 32 * tiles]
        operand["tile_dims"] = [tile_rows, 32]
        operand["format"] = data_format
        if operand["name"] != "out":
            operand["stimuli"] = {"seed": 16, "a": 2, "b": 4}[operand["name"]]
        else:
            operand["atol"] = operand["rtol"] = 0.0
    multiply = definition["operations"][0]["math"][1]
    multiply["math_fidelity"] = fidelity.name
    multiply["acc_to_dest"] = acc_to_dest
    config = FuserConfigSchema.load(
        f"elwmul_acc_{fidelity.name}_{acc_to_dest}_{data_format}_{dest_acc.name}_{tile_rows}_{dest_sync}",
        definition,
    )
    config.global_config.regenerate_cpp = True
    config.run_regular_test()
    result = torch.as_tensor(config.operand_registry.get("out").raw_data)
    expected = 24 if acc_to_dest else 8
    assert torch.all(result == expected), f"expected every element to be {expected}"


@pytest.mark.parametrize("first_accumulate", [False, True])
@pytest.mark.parametrize("dest_acc", [DestAccumulation.No, DestAccumulation.Yes])
def test_elwmul_reinit_acc_to_dest(first_accumulate, dest_acc):
    definition = FuserConfigSchema.load_definition("hifi_elwmul_overwrite")
    definition["dest_acc"] = dest_acc
    definition["loop_factor"] = 1
    for operand in definition["operands"]:
        if operand["name"] != "out":
            operand["stimuli"] = {"seed": 16, "a": 2, "b": 4}[operand["name"]]
        else:
            operand["atol"] = operand["rtol"] = 0.0
    math = definition["operations"][0]["math"]
    math[1]["acc_to_dest"] = first_accumulate
    math.append(deepcopy(math[1]))
    math[2]["acc_to_dest"] = not first_accumulate
    config = FuserConfigSchema.load(
        f"elwmul_reinit_acc_{first_accumulate}_{dest_acc.name}", definition
    )
    config.global_config.regenerate_cpp = True
    config.run_regular_test()
    result = torch.as_tensor(config.operand_registry.get("out").raw_data)
    expected = 8 if first_accumulate else 16
    assert torch.all(result == expected), f"expected every element to be {expected}"
