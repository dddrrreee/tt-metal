// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_device_operation_types.hpp"
#include "common/common.hpp"  // Data movement common utilities

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>

namespace ttnn::prim {

constexpr uint32_t BIT_MASK_32 = 32 - 1;

inline uint64_t ceil32(const uint64_t& number) {
    return ((number & BIT_MASK_32) == 0) ? number : ((number | BIT_MASK_32) + 1);
}

// maximal input/index/source/output chunk size, divisible by 32, calculated as follows:
// BH available L1 mem size of nearly 1.5 MB...
// ... divided by 4 to be able to allocate four equally long row chunks (coming from input/index/source/output
// tensors)
// ... divided by 4 to account for 4-byte datum sizes of each tensor (fp32, int32)
// ... minimized by ~10% to account for reserved memory
inline uint32_t calculate_optimal_chunk_size(const ttnn::Tensor& input_tensor) {
    uint32_t l1_per_chunk = (ttnn::operations::data_movement::get_max_l1_space(input_tensor) / 4) / 4;
    return ceil32((l1_per_chunk * 9 / 10) - 32);
}

// The writers write output row i to page i (get_noc_addr(i) + offset, up to the whole row), so each page must hold
// a whole row. A ROW_MAJOR output sharded narrower than a row has several pages per row: those writes land on
// other rows and, from the last row of a shard, past the end of the shard. The output given here is the device
// operation's, whose last axis is the scatter axis.
inline void validate_output_page_holds_row(const ttnn::Tensor& output_tensor) {
    const uint32_t row_bytes = output_tensor.logical_shape()[-1] * output_tensor.element_size();
    const uint32_t page_bytes = output_tensor.tensor_spec().compute_page_size_bytes();
    TT_FATAL(
        page_bytes >= row_bytes,
        "scatter: the {} output's shards are {} elements wide, narrower than its rows of {} elements along the "
        "scatter dim; use an output whose shards hold whole rows, for example interleaved or HEIGHT_SHARDED.",
        enchantum::to_string(output_tensor.memory_config().memory_layout()),
        page_bytes / output_tensor.element_size(),
        output_tensor.logical_shape()[-1]);
}

}  // namespace ttnn::prim
