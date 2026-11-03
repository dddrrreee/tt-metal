# Shard-placement diagnostics

Each script reproduces one ttnn op that finds "the core that holds shard i" by its own rule (a walk from core (0, 0), a device-wide row walk, absolute core coordinates, or the input's orientation for an output in another orientation) instead of following where the buffer placed the shards. The buffer places shard i on `corerange_to_cores(grid, n, orientation == ROW_MAJOR)[i]` (HEIGHT and WIDTH) or by position from the start of the single range (BLOCK), as in `BufferDistributionSpec::compute_core_list`. The analysis follows #63924; the cases are reported on #63678.

Run a script with `python tools/diagnostics/shard_walk/<script>.py` on one Blackhole chip, or on ttsim Blackhole (`TT_METAL_SIMULATOR`, `TT_METAL_SLOW_DISPATCH_MODE=1`, `TT_METAL_DISABLE_SFPLOADMACRO=1`). Each script prints one line per case: a control on a grid where both rules agree, then the grids where they do not. Values are chosen so that a correct result is exact in bfloat16, or the script states its tolerance. `conv2d_width_sharded.py` takes case indices, because two of its cases do not complete.

| script | op |
|---|---|
| `sdpa_decode_default_q.py` | `nlp_create_qkv_heads_decode` into `scaled_dot_product_attention_decode`, default memory configs |
| `sdpa_decode_sharded_q.py` | `scaled_dot_product_attention_decode`, sharded Q or output |
| `matmul_shard_grid.py` | `ttnn.matmul` / `ttnn.linear`: auto, 2D and 1D (mcast_in0, mcast_in1, gather_in0) configs |
| `nlp_create_qkv_heads.py` | `ttnn.experimental.nlp_create_qkv_heads`, WIDTH-sharded input |
| `nlp_create_qkv_heads_decode.py` | `ttnn.experimental.nlp_create_qkv_heads_decode`, output grid |
| `nlp_concat_heads_decode.py` | `ttnn.experimental.nlp_concat_heads_decode` |
| `nlp_kv_cache_load_slice.py` | `ttnn.experimental.nlp_kv_cache_load_slice`, preallocated output |
| `create_qkv_heads.py` | `ttnn.experimental.create_qkv_heads` |
| `create_qkv_heads_from_separate_tensors.py` | `ttnn.experimental.create_qkv_heads_from_separate_tensors` |
| `split_query_key_value_and_split_heads.py` | `ttnn.experimental.split_query_key_value_and_split_heads`, sharded |
| `rotary_embedding_llama.py` | `ttnn.experimental.rotary_embedding_llama`, prefill and decode |
| `group_attn_matmul.py` | `ttnn.experimental.group_attn_matmul` |
| `matmul_decode.py` | `ttnn.experimental.matmul_decode` |
| `group_norm_height_sharded.py` | `ttnn.group_norm`, HEIGHT_SHARDED |
| `fast_reduce_nc_nd_sharded.py` | `ttnn.experimental.fast_reduce_nc`, ND-sharded |
| `upsample_block_col_major.py` | `ttnn.upsample`, BLOCK_SHARDED COL_MAJOR |
| `untilize_with_unpadding_block.py` | `ttnn.untilize_with_unpadding`, BLOCK_SHARDED |
| `eltwise_output_orientation.py` | `ttnn.add` and `ttnn.where`, output in the other orientation |
| `bcast_h_sharded.py` | `ttnn.bcast(..., BcastOpDim.H)`, WIDTH and BLOCK sharded |
| `conv2d_width_sharded.py` | `ttnn.conv2d`, pre-sharded WIDTH_SHARDED input |
