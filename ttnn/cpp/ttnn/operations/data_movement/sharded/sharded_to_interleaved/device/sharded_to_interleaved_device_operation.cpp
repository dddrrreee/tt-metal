// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <algorithm>
#include <optional>
#include <vector>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// A tensor's pages as a grid of `rank` dims, outermost first: tiles for TILE, rows for ROW_MAJOR.
std::vector<uint64_t> page_grid(const ttnn::Shape& shape, const std::optional<Tile>& tile, size_t rank) {
    std::vector<uint64_t> grid(rank - shape.rank(), 1);
    grid.insert(grid.end(), shape.cbegin(), shape.cend());
    if (tile.has_value()) {
        grid[rank - 2] = tt::div_up(grid[rank - 2], uint64_t{tile->get_height()});
        grid[rank - 1] = tt::div_up(grid[rank - 1], uint64_t{tile->get_width()});
    } else {
        grid[rank - 1] = 1;
    }
    return grid;
}

// Page strides must agree wherever logical data spans multiple pages; this op also needs room for every padded page.
bool output_pages_line_up(const Tensor& input, const Tensor& output, bool every_padded_page) {
    const bool is_tile = input.layout() == Layout::TILE;
    const auto tile = is_tile ? std::optional<Tile>(input.tensor_spec().tile()) : std::nullopt;
    const auto& src_shape = is_tile ? input.padded_shape() : input.logical_shape();
    const size_t rank = std::max({size_t{2}, src_shape.rank(), output.padded_shape().rank()});
    const auto data = page_grid(input.logical_shape(), tile, rank);
    const auto src = page_grid(src_shape, tile, rank);
    const auto dst = page_grid(output.padded_shape(), tile, rank);
    uint64_t src_stride = 1;
    uint64_t dst_stride = 1;
    for (size_t i = data.size(); i-- > 0;) {
        if (data[i] > 1 && src_stride != dst_stride) {
            return false;
        }
        src_stride *= src[i];
        dst_stride *= dst[i];
    }
    return !every_padded_page || src_stride <= dst_stride;
}

}  // namespace

std::pair<bool, std::string> ShardedToInterleavedDeviceOperation::validate_inputs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;

    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return {false, "Operands to shard need to be on device!"};
    }
    if (!input_tensor.is_allocated()) {
        return {false, "Operands to shard need to be allocated in buffers on device!"};
    }
    if (!input_tensor.memory_config().is_sharded()) {
        return {false, "Input tensor must be sharded"};
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        if (output_tensor.memory_config() != args.output_mem_config) {
            return {false, "Mismatched output memory config"};
        }
        if (output_tensor.dtype() != args.output_dtype) {
            return {false, "Mismatched output dtype"};
        }
        if (output_tensor.storage_type() != StorageType::DEVICE) {
            return {false, "Operands to shard need to be on device!"};
        }
        if (!output_tensor.is_allocated()) {
            return {false, "Operands to shard need to be allocated in buffers on device!"};
        }
        if (output_tensor.device() != input_tensor.device()) {
            return {false, "Operands to shard need to be on the same device!"};
        }
        if (output_tensor.layout() != input_tensor.layout()) {
            return {false, "Output tensor layout must match input tensor layout"};
        }
        // Errors, not {false, ...}: this op and its ttnn.copy fallback both write input page n to output page n.
        TT_FATAL(
            output_tensor.logical_shape() == input_tensor.logical_shape(),
            "Input tensor shape {} does not match output tensor shape {}",
            input_tensor.logical_shape(),
            output_tensor.logical_shape());
        const bool same_tile = input_tensor.layout() != Layout::TILE ||
                               output_tensor.tensor_spec().tile() == input_tensor.tensor_spec().tile();
        if (args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED && same_tile) {
            TT_FATAL(
                output_pages_line_up(input_tensor, output_tensor, false),
                "Output tensor padded shape {} lays out pages differently from input tensor page layout {} for "
                "shape {}",
                output_tensor.padded_shape(),
                input_tensor.layout() == Layout::TILE ? input_tensor.padded_shape() : input_tensor.logical_shape(),
                input_tensor.logical_shape());
        }
    }
    if (input_tensor.memory_config().buffer_type() != BufferType::L1) {
        return {false, "Input tensor must be in L1"};
    }
    if (args.output_mem_config.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return {false, "Output memory config must be Interleaved"};
    }
    if (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED) {
        return {
            false,
            "sharded_to_interleaved does not support ND sharding. Please use ttnn.to_memory_config or ttnn.copy "
            "instead."};
    }
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        uint32_t l1_alignment = hal::get_l1_alignment();
        if ((*input_tensor.memory_config().shard_spec()).shape[1] * input_tensor.element_size() % l1_alignment != 0) {
            return {false, fmt::format("Shard page size must be aligned to {}B for L1 Tensor", l1_alignment)};
        }
    }
    if (input_tensor.dtype() != args.output_dtype) {
        if (input_tensor.layout() != Layout::TILE) {
            return {false, "If diff output type, tensor must be TILED"};
        }
    }
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {
                false,
                fmt::format(
                    "sharded_to_interleaved requires standard 32x32 tiles, got {}x{}",
                    tile.get_height(),
                    tile.get_width())};
        }
        if (tensor_args.preallocated_output.has_value()) {
            auto out_tile = tensor_args.preallocated_output.value().tensor_spec().tile();
            if (out_tile != tile) {
                return {false, "Output tensor tile shape must match input tensor tile shape"};
            }
        }
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto& output_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            output_pages_line_up(input_tensor, output_tensor, true),
            "Output tensor padded shape {} has fewer pages than input tensor padded shape {}",
            output_tensor.padded_shape(),
            input_tensor.padded_shape());
    }
    return {true, ""};
}

void ShardedToInterleavedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    auto [valid, msg] = validate_inputs(args, tensor_args);
    TT_FATAL(valid, "{}", msg);
}

tt::tt_metal::TensorSpec ShardedToInterleavedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    // An interleaved row-major page is one logical row. Plain TensorLayout means alignment 1, so
    // padding lands only in aligned_page_size; the factory clips each write to match.
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        return tt::tt_metal::TensorSpec(
            input_tensor.logical_shape(),
            tt::tt_metal::TensorLayout(
                args.output_dtype, tt::tt_metal::PageConfig(input_tensor.layout()), args.output_mem_config));
    }

    // Tile keeps the padded shape: the writer indexes off that same padded row, so both must agree.
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout::fromPaddedShape(
            args.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            args.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

Tensor ShardedToInterleavedDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto spec = compute_output_specs(args, tensor_args);
    return create_device_tensor(spec, input_tensor.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor>
ShardedToInterleavedDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) const {
    int ideal_dev_clock_cycles = operations::data_movement::common_tm_bw_model(tensor_args.input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {tensor_args.input_tensor}, {output_tensor}, ideal_dev_clock_cycles);
    return result;
}

Tensor sharded_to_interleaved(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::DataType& output_dtype,
    const std::optional<Tensor>& preallocated_output) {
    return ttnn::device_operation::launch<ShardedToInterleavedDeviceOperation>(
        ShardedToInterleavedParams{output_mem_config, output_dtype, 1, 0},
        ShardedToInterleavedInputs{input_tensor, preallocated_output});
}

}  // namespace ttnn::prim
