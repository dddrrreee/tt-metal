# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# The large cases of test_unary_dram_reads.py: tensors of about 600 pages per core over the whole grid (2100 for the very
# large interleaved case), most of the data that suite pushes through each op. They live apart so the simulator legs can
# skip this file (tests/pipeline_reorg/ttsim-skip-list.yaml) and still run the small cases, which take every flow.

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.eltwise import test_unary_dram_reads as base
from tests.ttnn.unit_tests.operations.eltwise.test_unary_dram_reads import (
    DEFAULT_PATH_GRID,
    OPS,
    OTHER_FORMATS_OPS,
    large_rows,
    memory_config_for,
    random_input,
    run_against_default_path,
)


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, num_shards",
    [
        ("large", 7),
        ("large_four_shards_short_last", 4),
        ("large_all_banks", None),
    ],
)
def test_height_sharded(device, op_name, case, num_shards):
    base.test_height_sharded(device, op_name, case, num_shards)


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize(
    "case, width, num_shards",
    [
        ("large", 7 * 1024, 7),
        ("large_one_tile_shards", 7 * 32, 7),
        ("large_four_shards", 4 * 1024, 4),
        ("large_all_banks", None, None),
    ],
)
def test_width_sharded(device, op_name, case, width, num_shards):
    base.test_width_sharded(device, op_name, case, width, num_shards)


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("case", ["large"])
def test_interleaved(device, op_name, case):
    base.test_interleaved(device, op_name, case)


# Light ops take the queue on very large tensors too. Compared on device, as host conversion of these tensors is slow.
@pytest.mark.parametrize("op_name", ["identity", "relu", "hardswish"])
def test_interleaved_very_large(device, op_name):
    shape = [1, 1, large_rows(device, 1024, 32, pages_per_core=2100), 1024]
    x = ttnn.rand(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, low=0.05, high=0.95)
    out = OPS[op_name](x)
    reference = OPS[op_name](x, sub_core_grids=DEFAULT_PATH_GRID)
    differing = ttnn.sum(ttnn.typecast(ttnn.ne(out, reference), ttnn.float32))
    assert ttnn.to_torch(differing).item() == 0


@pytest.mark.parametrize(
    "dtype, size",
    [
        (ttnn.float32, "large"),
        (ttnn.bfloat8_b, "large"),
    ],
    ids=["fp32-large", "bfp8-large"],
)
@pytest.mark.parametrize("op_name", OTHER_FORMATS_OPS)
@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_other_formats(device, dtype, op_name, layout, size):
    base.test_other_formats(device, dtype, op_name, layout, size)


@pytest.mark.parametrize("layout", ["height", "width", "interleaved"])
def test_int32_relu(device, layout):
    width = 7 * 1024 if layout == "width" else 1024
    shape = [1, 1, large_rows(device, width, 32 * 7), width]
    torch.manual_seed(0)
    a = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    run_against_default_path(ttnn.relu, a, device, memory_config_for(layout, shape), ttnn.int32)


# Two queue shapes share one program: the cache-hit path must rewrite the total page count and, on shards, the last
# shard's size. Growing on the hit catches a stale smaller count, which would leave the last pages unwritten.
@pytest.mark.parametrize("layout", ["height", "interleaved"])
@pytest.mark.parametrize("grow", [False, True], ids=["shrink", "grow"])
def test_work_queue_program_cache_reuse(device, layout, grow):
    device.cache_entries_counter.reset()
    large = large_rows(device, 1024, 32 * 7)
    for rows in (large - 64, large) if grow else (large, large - 64):
        shape = [1, 1, rows, 1024]
        run_against_default_path(
            ttnn.relu,
            random_input(shape, seed=rows),
            device,
            memory_config_for(layout, shape),
            measure=device.cache_entries_counter.measure,
        )
    assert device.cache_entries_counter.total == 1
