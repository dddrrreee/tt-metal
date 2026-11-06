# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Fast post-commit tests for SDPA decode with a representative subset of parametrizations.
"""

import random

import torch
import numpy as np
import pytest
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import (
    num_to_corerange,
    run_test_sdpa_decode_single_iter,
    run_test_sdpa_decode_multi_pos,
    run_test_sdpa_decode_paged_attention,
    run_test_sdpa_decode_broadcast_mask_batch,
)


@pytest.fixture(scope="function")
def reset_seeds():
    torch.manual_seed(213919)
    np.random.seed(213919)
    random.seed(213919)
    yield


def test_sdpa_decode_fp32_half_sync_cross_core_reduction(device):
    """#56171: merging two KV chunks used five DST slots, but FP32 half-sync has four."""
    torch.manual_seed(0)
    heads, kv_heads, head_dim, cache = 8, 2, 128, 128
    q = torch.randn(1, 1, heads, head_dim)
    k = torch.randn(1, kv_heads, cache, head_dim)
    v = torch.randn(1, kv_heads, cache, head_dim)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(2, 2),
        q_chunk_size=32,
        k_chunk_size=32,
        max_cores_per_head_batch=2,  # Two cores per KV head force cross-core correction at position 32.
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    # Position 31 stays within one chunk; position 32 activates the second core
    # and previously corrupted the result in the fused softmax correction.
    for cur_pos in (31, 32):
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            tq,
            tk,
            tv,
            cur_pos=[cur_pos],
            is_causal=True,
            program_config=program_config,
            compute_kernel_config=compute_config,
        )
        valid_k = k[:, :, : cur_pos + 1].repeat_interleave(heads // kv_heads, dim=1)
        valid_v = v[:, :, : cur_pos + 1].repeat_interleave(heads // kv_heads, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3), valid_k, valid_v)
        actual = ttnn.to_torch(out)[:, :, :heads]
        assert_with_pcc(ref.permute(0, 2, 1, 3), actual, 0.999)
        ttnn.deallocate(out)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    (
        [8, 8, 1, 32768, 128, (8, 6), True, False],  # Llama2-70B
        [4, 32, 8, 8192, 128, (8, 8), True, True],  # llama 3.1 8b
    ),
)
@pytest.mark.timeout(120)
def test_sdpa_decode(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    if single_iter:
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )
    else:
        run_test_sdpa_decode_multi_pos(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
        )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor",
    ([2, 20, 20, 512, 64, (8, 8), True],),  # Whisper-large (nh not multiple of 32; grid must give num_cores/b <= nkv)
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_tile_aligned_heads(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor):
    """Regression test for models with num_heads not a multiple of 32 (e.g. Whisper-large with 20 heads).

    The output logical shape must preserve the unpadded head count so that downstream
    ops like nlp_concat_heads produce the correct hidden dimension.
    """
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 64, 8, 2048, 128, (8, 8)],),  # num q heads greater than 32
)
@pytest.mark.timeout(120)
def test_sdpa_decode_non_causal(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    for _ in range(2):
        run_test_sdpa_decode_single_iter(
            device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=False, causal=False
        )
    assert device.cache_entries_counter.total == 1


@pytest.mark.parametrize("num_chunks", [1, 2, 3], ids=["inactive-core", "one-per-core", "multiple-per-core"])
def test_sdpa_decode_non_causal_chunk_distribution(device, num_chunks):
    """Exercise both sides of the single-local-chunk specialization with two cores per KV head."""
    torch.manual_seed(1234)
    heads, kv_heads, head_dim, chunk_size = 8, 2, 64, 32
    q = torch.randn(1, 1, heads, head_dim)
    k = torch.randn(1, kv_heads, num_chunks * chunk_size, head_dim)
    v = torch.randn_like(k)
    tq, tk, tv = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in (q, k, v))
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq,
        tk,
        tv,
        is_causal=False,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(2, 2),
            q_chunk_size=32,
            k_chunk_size=chunk_size,
            max_cores_per_head_batch=2,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        ),
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.repeat_interleave(heads // kv_heads, dim=1),
        v.repeat_interleave(heads // kv_heads, dim=1),
    )
    assert_with_pcc(ref.permute(0, 2, 1, 3), ttnn.to_torch(out), 0.999)


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat16, ttnn.bfloat16],
    ],
    ids=[
        "all_bfp16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, single_iter, cur_pos_tensor",
    ([32, 8, 1, 32768, 128, (8, 6), True, True],),  # Llama2-70B
)
def test_sdpa_decode_ignore_users(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, single_iter, cur_pos_tensor):
    # Set odd users to -1 to test skipping users
    start_indices = [100 if bb % 2 == 0 else -1 for bb in range(b)]

    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        sharded_in=False,
        sharded_out=False,
        start_indices=start_indices,
    )


@pytest.mark.parametrize(
    "kv_dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, cur_pos_tensor, sliding_window_size",
    ([8, 16, 4, 4096, 128, (8, 2), True, None],),  # llama 3.1 8b N300
    ids=["llama3.1-a"],
)
@pytest.mark.parametrize("block_size", (64,), ids=["paged_64"])
def test_sdpa_decode_paged_attention(
    device, b, nh, nkv, s, d, kv_dtype, grid_size, q_dtype, cur_pos_tensor, sliding_window_size, block_size, reset_seeds
):
    if s == 128 * 1024 and block_size != 64:
        # 128k sequence, block_size 64 tests the sizing of the page table CB
        pytest.skip("Skipping test for seq_len=128k with block_size!=64")
    run_test_sdpa_decode_paged_attention(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        kv_dtype,
        grid_size,
        q_dtype,
        cur_pos_tensor,
        block_size=block_size,
        sharded_in=True,
        sharded_out=False,
        sliding_window_size=sliding_window_size,
    )

    assert device.num_program_cache_entries() == 4


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    [
        [32, 32, 8, 2048, 128, (10, 11)],
    ],
    ids=["blackhole_b32_nkv8"],
)
@pytest.mark.timeout(120)
def test_sdpa_decode_kv_head_core_divisibility(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    """Regression test for github.com/tenstorrent/tt-metal/issues/40978.

    When floor(num_cores_available / B) yields a value that makes ceil(num_kv_heads / uncapped)
    a non-divisor of num_kv_heads, the old core allocation produced inconsistent counts causing
    a TT_FATAL crash at output core indexing.
    """
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, cur_pos_tensor=True, sharded_in=False, sharded_out=False
    )


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat8_b],
    ],
    ids=[
        "all_bfp8",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size",
    ([1, 8, 1, 32768, 128, (8, 8)],),
)
def test_sdpa_decode_sharded(device, b, nh, nkv, s, d, dtype, grid_size, q_dtype):
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=False
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=True, sharded_out=True
    )
    run_test_sdpa_decode_single_iter(
        device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sharded_in=False, sharded_out=True
    )


@pytest.mark.parametrize("dtype, q_dtype", [[ttnn.bfloat8_b, ttnn.bfloat16]], ids=["kv_bfp8_q_bf16"])
@pytest.mark.parametrize("b, nh, nkv, s, d", ([32, 32, 8, 2048, 128], [8, 32, 8, 2048, 128]))
def test_sdpa_decode_sharded_q_block_wide_grid(device, b, nh, nkv, s, d, dtype, q_dtype):
    # The Q shard stays on the 8x4 block used by the model code while the op runs on the whole grid.
    grid = device.compute_with_storage_grid_size()
    if grid.x <= 8 or grid.y <= 4:
        pytest.skip(f"needs a grid wider than 8 and taller than 4, got {grid}")
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(8, 0), ttnn.CoreCoord(grid.x - 1, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(grid.x - 1, grid.y - 1)),
        ]
    )
    run_test_sdpa_decode_single_iter(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        dtype,
        (grid.x, grid.y),
        q_dtype,
        sharded_in=True,
        sharded_out=False,
        sub_core_grids=sub_core_grids,
    )


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b],
    ids=["bfp8"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([16, 8, 1, 8192, 128],),  # Llama2-70B
)
def test_sdpa_decode_program_cache(device, b, nh, nkv, s, d, dtype, reset_seeds):
    dummy_tensors = []
    # One cur_pos vector for both outer passes: compute_program_hash includes cur_pos, so resampling
    # per iteration would compile extra programs for cur_pos_tensor=False paths (expected cache size 4).
    start_indices = np.random.randint(0, s - 1, b).tolist()
    start_indices[0] = s - 1
    for i in range(2):
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(32, 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )
        dummy_tensors.append(
            ttnn.as_tensor(
                torch.zeros(1, 1, 32, 32 * 32),
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet({num_to_corerange(32)}),
                        (32, 32),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
            )
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=False,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=True,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=False,
        )
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            (8, 6),
            dtype,
            sharded_in=False,
            sharded_out=True,
            start_indices=start_indices,
            cur_pos_tensor=True,
        )

    assert device.num_program_cache_entries() == 4


@pytest.mark.parametrize(
    "dtype, q_dtype",
    [
        [ttnn.bfloat8_b, ttnn.bfloat16],
    ],
    ids=[
        "kv_bfp8_q_bf16",
    ],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, grid_size, sliding_window_size",
    [
        [4, 8, 1, 1024, 128, (8, 4), 128],  # Medium window
    ],
)
@pytest.mark.parametrize("cur_pos_tensor", [True], ids=["cur_pos_tensor"])
@pytest.mark.timeout(120)
def test_sdpa_decode_sliding_window(
    device, b, nh, nkv, s, d, dtype, grid_size, q_dtype, sliding_window_size, cur_pos_tensor
):
    """Test sliding window attention functionality."""

    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")

    # Ensure sliding window is smaller than sequence length
    if sliding_window_size >= s:
        pytest.skip(f"Sliding window {sliding_window_size} must be smaller than sequence length {s}")

    # Test different window start positions to ensure all fill tile code paths are hit
    # when generating the sliding window
    k_values = [5, 10, 20, 30]

    test_positions = [
        *[
            (sliding_window_size + offset - 1) + 32 * k
            for k in k_values
            for offset in (15, 16, 17)
            # in first face, in second face (1st face completely filled), in second face (1st face completely filled + 2nd face partially filled)
        ],
        sliding_window_size * 2,
        sliding_window_size // 2,
        sliding_window_size - 1,
        s // 2,
        s - 10,
    ]
    for cur_pos in test_positions:
        if cur_pos >= s:
            continue

        # Test both cur_pos and cur_pos_tensor modes
        run_test_sdpa_decode_single_iter(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            dtype,
            grid_size,
            q_dtype,
            cur_pos_tensor=cur_pos_tensor,
            sharded_in=False,
            sharded_out=False,
            start_indices=[cur_pos + i for i in range(b)],  # test a batch with different start positions
            sliding_window_size=sliding_window_size,
        )


@pytest.mark.parametrize("mask_dtype", [ttnn.bfloat16, ttnn.bfloat4_b])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, dtype, grid_size",
    [
        (32, 8, 1, 2048, 128, ttnn.bfloat8_b, (8, 8)),
    ],
)
def test_sdpa_decode_broadcast_mask_batch(device, b, nh, nkv, s, d, dtype, grid_size, mask_dtype):
    run_test_sdpa_decode_broadcast_mask_batch(
        device,
        b=b,
        nh=nh,
        nkv=nkv,
        s=s,
        d=d,
        dtype=dtype,
        grid_size=grid_size,
        mask_dtype=mask_dtype,
    )


def _height_sharded_memcfg(first, last, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*first), ttnn.CoreCoord(*last))})
    spec = ttnn.ShardSpec(grid, shard_shape, orientation)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


@pytest.mark.parametrize(
    "paged, q_shard, out_shard, grid_size, expected_error",
    [
        (
            False,
            ((0, 1), (3, 1), (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (4, 2),
            "Sharded Q shard 0 is on core .*0.*1.*, but batch 0 runs on core .*0.*0",
        ),
        (
            False,
            ((0, 0), (3, 0), (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (2, 2),
            "Sharded Q shard 2 is on core .*2.*0.*, but batch 2 runs on core .*0.*1",
        ),
        (
            False,
            ((0, 0), (1, 1), (32, 128), ttnn.ShardOrientation.COL_MAJOR),
            None,
            (2, 2),
            "Sharded Q shard 1 is on core .*0.*1.*, but batch 1 runs on core .*1.*0",
        ),
        (
            False,
            ((0, 0), (1, 0), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (4, 2),
            "Sharded Q has 2 shard cores, but batch size 4 requires at least 4 shard cores",
        ),
        (
            False,
            None,
            ((0, 1), (3, 1), (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
            (4, 2),
            "Sharded output shard 0 is on core .*0.*1.*, but batch 0 runs on core .*0.*0",
        ),
        (
            True,
            ((1, 0), (4, 0), (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (4, 2),
            "Sharded Q shard 0 is on core .*1.*0.*, but batch 0 runs on core .*0.*0",
        ),
        (
            False,
            ((0, 0), (3, 0), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (4, 2),
            "Sharded Q shard height 64 must hold one batch of 32 padded heads without sharing a shard",
        ),
        (
            True,
            ((0, 0), (3, 0), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
            None,
            (4, 2),
            "Sharded Q shard height 64 must hold one batch of 32 padded heads without sharing a shard",
        ),
        (
            False,
            None,
            ((0, 0), (3, 0), (64, 128), ttnn.ShardOrientation.ROW_MAJOR),
            (4, 2),
            "Sharded output shard height 64 must hold one batch of 32 padded heads without sharing a shard",
        ),
    ],
    ids=[
        "q_shifted",
        "q_wrapped",
        "q_col_major",
        "q_too_few_cores",
        "out_shifted",
        "paged_q_shifted",
        "q_multi_batch",
        "paged_q_multi_batch",
        "out_multi_batch",
    ],
)
def test_sdpa_decode_rejects_misplaced_shards(
    device, expect_error, paged, q_shard, out_shard, grid_size, expected_error
):
    dev_grid = device.compute_with_storage_grid_size()
    if dev_grid.x < 5 or dev_grid.y < 2:
        pytest.skip(f"Need at least 5x2 compute grid, got {dev_grid}")

    torch.manual_seed(0)
    batch, heads, kv_heads, seq_len, head_dim = 4, 8, 1, 128, 128
    q = torch.randn(1, batch, heads, head_dim, dtype=torch.bfloat16)
    q_memcfg = _height_sharded_memcfg(*q_shard) if q_shard is not None else ttnn.DRAM_MEMORY_CONFIG
    out_memcfg = _height_sharded_memcfg(*out_shard) if out_shard is not None else ttnn.DRAM_MEMORY_CONFIG
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=q_memcfg)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid_size),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    if paged:
        block_size = 64
        num_blocks_per_seq = seq_len // block_size
        k = torch.randn(batch * num_blocks_per_seq, kv_heads, block_size, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch * num_blocks_per_seq, kv_heads, block_size, head_dim, dtype=torch.bfloat16)
        page_table = torch.arange(batch * num_blocks_per_seq, dtype=torch.int32).reshape(batch, num_blocks_per_seq)
        cur_pos = torch.tensor([63, 79, 95, 111], dtype=torch.int32)
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        t_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        t_pos = ttnn.from_torch(cur_pos, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        with expect_error(RuntimeError, expected_error):
            ttnn.transformer.paged_scaled_dot_product_attention_decode(
                tq,
                tk,
                tv,
                t_pt,
                cur_pos_tensor=t_pos,
                program_config=program_config,
                memory_config=out_memcfg,
            )
    else:
        k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
        v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
        tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        with expect_error(RuntimeError, expected_error):
            ttnn.transformer.scaled_dot_product_attention_decode(
                tq,
                tk,
                tv,
                cur_pos=[63, 79, 95, 111],
                program_config=program_config,
                memory_config=out_memcfg,
            )


@pytest.mark.parametrize("heads, shard_height, end", [(8, 64, (3, 0)), (160, 96, (3, 1))])
def test_mla_decode_rejects_multi_batch_q_shard(device, expect_error, heads, shard_height, end):
    q = torch.ones((1, 4, heads, 128), dtype=torch.bfloat16)
    k = torch.ones((4, 1, 128, 128), dtype=torch.bfloat16)
    memcfg = _height_sharded_memcfg((0, 0), end, (shard_height, 128))
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    positions = ttnn.from_torch(torch.tensor([63, 79, 95, 111], dtype=torch.int32), device=device, dtype=ttnn.int32)
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 2), q_chunk_size=32, k_chunk_size=32
    )
    with expect_error(RuntimeError, f"Sharded Q shard height {shard_height} must hold one batch"):
        ttnn.transformer.flash_multi_latent_attention_decode(
            tq, tk, tk, head_dim_v=128, cur_pos_tensor=positions, program_config=config
        )


def test_mla_decode_rejects_uneven_head_group_shards(device, expect_error):
    """Two 64-row head groups cannot split 96 heads; heads 32..63 would be skipped."""
    q = torch.randn((1, 1, 96, 128), dtype=torch.bfloat16)
    k = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)
    memcfg = _height_sharded_memcfg((0, 0), (1, 0), (64, 128))
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    positions = ttnn.from_torch(torch.tensor([63], dtype=torch.int32), device=device, dtype=ttnn.int32)
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 2), q_chunk_size=32, k_chunk_size=32
    )
    with expect_error(RuntimeError, "Sharded Q shard height 64 must hold one batch of 96 padded heads"):
        ttnn.transformer.flash_multi_latent_attention_decode(
            tq, tk, tk, head_dim_v=128, cur_pos_tensor=positions, program_config=config
        )


def test_sdpa_decode_accepts_single_batch_oversized_shards(device):
    q = torch.ones((1, 1, 8, 128), dtype=torch.bfloat16)
    k = torch.ones((1, 1, 128, 128), dtype=torch.bfloat16)
    v = torch.ones_like(k)
    memcfg = _height_sharded_memcfg((0, 0), (0, 0), (64, 128))
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 2), q_chunk_size=32, k_chunk_size=32
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq, tk, tv, cur_pos=[63], program_config=config, memory_config=memcfg
    )
    torch.testing.assert_close(ttnn.to_torch(out).float(), torch.ones_like(q).float(), rtol=0, atol=0.02)


@pytest.mark.parametrize(
    "q_range, out_range, grid_size, sub_range",
    [
        (((0, 0), (3, 0)), ((0, 0), (3, 0)), (4, 2), None),
        (((0, 0), (1, 1)), ((0, 0), (1, 1)), (2, 2), None),
        (((0, 0), (5, 0)), None, (4, 2), None),
        (((0, 1), (3, 1)), ((0, 1), (3, 1)), (4, 2), ((0, 1), (3, 2))),
    ],
    ids=["aligned_row", "wrapped_2x2", "spare_shard_cores", "sub_core_grids_offset"],
)
def test_sdpa_decode_accepts_placed_shards(device, q_range, out_range, grid_size, sub_range):
    dev_grid = device.compute_with_storage_grid_size()
    if dev_grid.x < 6 or dev_grid.y < 3:
        pytest.skip(f"Need at least 6x3 compute grid, got {dev_grid}")

    torch.manual_seed(0)
    batch, heads, kv_heads, seq_len, head_dim = 4, 8, 1, 128, 128
    cur_pos = [63, 79, 95, 111]
    q = torch.randn(1, batch, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16)

    q_memcfg = _height_sharded_memcfg(*q_range, (32, head_dim))
    out_memcfg = (
        _height_sharded_memcfg(*out_range, (32, head_dim)) if out_range is not None else ttnn.DRAM_MEMORY_CONFIG
    )
    sub_core_grids = (
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*sub_range[0]), ttnn.CoreCoord(*sub_range[1]))})
        if sub_range is not None
        else None
    )
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=q_memcfg)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid_size),
        sub_core_grids=sub_core_grids,
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
        dst_full_sync_en=False,
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq,
        tk,
        tv,
        cur_pos=cur_pos,
        program_config=program_config,
        compute_kernel_config=compute_config,
        memory_config=out_memcfg,
    )
    actual = ttnn.to_torch(out)[:, :, :heads].float()
    expected = torch.empty_like(actual)
    for b, pos in enumerate(cur_pos):
        valid_k = k[b : b + 1, :, : pos + 1].repeat_interleave(heads // kv_heads, dim=1).float()
        valid_v = v[b : b + 1, :, : pos + 1].repeat_interleave(heads // kv_heads, dim=1).float()
        expected[0, b] = torch.nn.functional.scaled_dot_product_attention(
            q[:, b, :heads].unsqueeze(2).float(), valid_k, valid_v
        )[0, :, 0]
    assert_with_pcc(expected, actual, 0.999)
    ttnn.deallocate(out)


def test_sdpa_decode_row_major_q_sharded_output_multi_batch(device):
    """Row-major Q in 8-row shards with tile-height output shards, as Galaxy decode uses: batch b lands on core b."""
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 8, 8, 128, 128
    cur_pos = [15 + 8 * b for b in range(batch)]
    q = torch.randn(1, batch, heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch, 1, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch, 1, seq_len, head_dim, dtype=torch.bfloat16)
    q_memcfg = _height_sharded_memcfg((0, 0), (7, 0), (heads, head_dim))
    out_memcfg = _height_sharded_memcfg((0, 0), (7, 0), (32, head_dim))
    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=q_memcfg)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 2), q_chunk_size=32, k_chunk_size=32
    )
    out = ttnn.transformer.scaled_dot_product_attention_decode(
        tq, tk, tv, cur_pos=cur_pos, program_config=program_config, memory_config=out_memcfg
    )
    # The kernel writes batch b to the first rows of core b's 32-row shard, which the [1, 8, 8, 128] output does not
    # map; read every shard through a row-major tensor of 8 full shards allocated at the same address.
    address = out.buffer_address()
    ttnn.deallocate(out)
    shards = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, batch * 32, head_dim]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, out_memcfg
    )
    assert shards.buffer_address() == address
    actual = ttnn.to_torch(shards).float().reshape(1, batch, 32, head_dim)[:, :, :heads]
    expected = torch.empty_like(actual)
    for b, pos in enumerate(cur_pos):
        valid_k = k[b : b + 1, :, : pos + 1].repeat_interleave(heads, dim=1).float()
        valid_v = v[b : b + 1, :, : pos + 1].repeat_interleave(heads, dim=1).float()
        expected[0, b] = torch.nn.functional.scaled_dot_product_attention(
            q[:, b].unsqueeze(2).float(), valid_k, valid_v
        )[0, :, 0]
    assert_with_pcc(expected, actual, 0.999)
