// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ttnn::to_memory_config from a sharded tensor into a preallocated interleaved output whose padded shape
// differs from the input's. Only explicit alignments make these shapes, so the cases live here, not in Python.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

namespace ttnn::test {
namespace {

using ::testing::HasSubstr;
using ::testing::ThrowsMessage;
using ::tt::tt_metal::Alignment;
using ::tt::tt_metal::BufferType;
using ::tt::tt_metal::CoreCoord;
using ::tt::tt_metal::CoreRange;
using ::tt::tt_metal::CoreRangeSet;
using ::tt::tt_metal::DataType;
using ::tt::tt_metal::Layout;
using ::tt::tt_metal::MemoryConfig;
using ::tt::tt_metal::PageConfig;
using ::tt::tt_metal::ShardOrientation;
using ::tt::tt_metal::ShardSpec;
using ::tt::tt_metal::TensorLayout;
using ::tt::tt_metal::TensorMemoryLayout;
using ::tt::tt_metal::TensorSpec;

const ttnn::Shape kShape({1, 1, 64, 64});

// uint32 [1, 1, 64, 64] holding 1..4096, padded to [1, 1, 64, 96] and height-sharded on two cores ([32, 96]).
Tensor make_width_padded_sharded_input(tt::tt_metal::distributed::MeshDevice* device, std::vector<uint32_t>& data) {
    data.resize(kShape.volume());
    std::iota(data.begin(), data.end(), 1u);
    const MemoryConfig sharded(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0))), {32, 96}, ShardOrientation::ROW_MAJOR));
    const TensorSpec spec(kShape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), sharded, Alignment({32, 96})));
    return Tensor::from_vector(data, spec, device);
}

Tensor make_dram_output(tt::tt_metal::distributed::MeshDevice* device, const Alignment& alignment) {
    const TensorSpec spec(
        kShape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG, alignment));
    return Tensor::from_vector(std::vector<uint32_t>(kShape.volume(), 0), spec, device);
}

// The output has the input's tile row pitch (3 tiles) and one more tile row of padding. The op writes every
// input tile to the same tile index of the output, which is where the output keeps it.
TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedOutputWithTallerPadding) {
    std::vector<uint32_t> data;
    const Tensor input = make_width_padded_sharded_input(device_, data);
    const Tensor output = make_dram_output(device_, Alignment({96, 96}));
    ASSERT_EQ(input.padded_shape(), ttnn::Shape({1, 1, 64, 96}));
    ASSERT_EQ(output.padded_shape(), ttnn::Shape({1, 1, 96, 96}));

    ttnn::to_memory_config(input, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, output);
    EXPECT_EQ(output.to_vector<uint32_t>(), data);
}

// The output's padded shape has a higher rank than its logical shape. The extra dims are 1, so pages line up.
TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedOutputWithHigherRankPadding) {
    const ttnn::Shape shape({64, 64});
    std::vector<uint32_t> data(shape.volume());
    std::iota(data.begin(), data.end(), 1u);
    const MemoryConfig sharded(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0))), {32, 64}, ShardOrientation::ROW_MAJOR));
    const TensorSpec input_spec(shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), sharded));
    const TensorSpec output_spec(
        shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG, Alignment({1, 1, 96, 64})));
    const Tensor input = Tensor::from_vector(data, input_spec, device_);
    const Tensor output = Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 0), output_spec, device_);
    ASSERT_EQ(output.padded_shape(), ttnn::Shape({1, 1, 96, 64}));

    ttnn::to_memory_config(input, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, output);
    EXPECT_EQ(output.to_vector<uint32_t>(), data);
}

// An output with a 2-tile row pitch cannot hold the input's 3-tile rows in place, and would take 6 tiles into 4.
TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedRejectsNarrowerPaddedOutput) {
    std::vector<uint32_t> data;
    const Tensor input = make_width_padded_sharded_input(device_, data);
    const Tensor output = make_dram_output(device_, Alignment{});
    ASSERT_EQ(output.padded_shape(), kShape);

    EXPECT_THAT(
        [&] { ttnn::to_memory_config(input, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, output); },
        ThrowsMessage<std::runtime_error>(HasSubstr("lays out pages differently")));
}

// Only the capacity check can reject this: the data occupies one tile, but the input has four padded tiles.
TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedRejectsInsufficientPageCapacity) {
    const ttnn::Shape shape({1, 1, 32, 32});
    const MemoryConfig sharded(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::L1,
        ShardSpec(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), {64, 64}, ShardOrientation::ROW_MAJOR));
    const TensorSpec input_spec(shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), sharded, Alignment({64, 64})));
    const TensorSpec output_spec(shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    const Tensor input = Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 1), input_spec, device_);
    const Tensor output = Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 0), output_spec, device_);
    ASSERT_EQ(input.padded_shape(), ttnn::Shape({1, 1, 64, 64}));
    ASSERT_EQ(output.padded_shape(), shape);
    EXPECT_THAT(
        [&] {
            (void)ttnn::prim::ShardedToInterleavedDeviceOperation::validate_inputs(
                {ttnn::DRAM_MEMORY_CONFIG, DataType::UINT32}, {input, output});
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("has fewer pages")));
}

// The same input in DRAM falls back to ttnn.copy, which copies only the one data tile, so this output works.
TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedCopyFallbackIgnoresInputPaddingPages) {
    const ttnn::Shape shape({1, 1, 32, 32});
    std::vector<uint32_t> data(shape.volume());
    std::iota(data.begin(), data.end(), 1u);
    const MemoryConfig sharded(
        TensorMemoryLayout::HEIGHT_SHARDED,
        BufferType::DRAM,
        ShardSpec(CoreRangeSet(CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))), {64, 64}, ShardOrientation::ROW_MAJOR));
    const TensorSpec input_spec(
        shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), sharded, Alignment({64, 64})));
    const TensorSpec output_spec(
        shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    const Tensor input = Tensor::from_vector(data, input_spec, device_);
    const Tensor output = Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 0), output_spec, device_);
    ASSERT_EQ(input.padded_shape(), ttnn::Shape({1, 1, 64, 64}));

    ttnn::to_memory_config(input, ttnn::DRAM_MEMORY_CONFIG, std::nullopt, output);
    EXPECT_EQ(output.to_vector<uint32_t>(), data);
}

TEST_F(TTNNFixtureWithDevice, ToMemoryConfigShardedToInterleavedDirectValidationRejectsOtherShape) {
    std::vector<uint32_t> data;
    const Tensor input = make_width_padded_sharded_input(device_, data);
    const ttnn::Shape shape({1, 1, 64, 32});
    const TensorSpec output_spec(shape, TensorLayout(DataType::UINT32, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    const Tensor output = Tensor::from_vector(std::vector<uint32_t>(shape.volume(), 0), output_spec, device_);
    EXPECT_THAT(
        [&] {
            (void)ttnn::prim::ShardedToInterleavedDeviceOperation::validate_inputs(
                {ttnn::DRAM_MEMORY_CONFIG, DataType::UINT32}, {input, output});
        },
        ThrowsMessage<std::runtime_error>(HasSubstr("does not match output tensor shape")));
}

}  // namespace
}  // namespace ttnn::test
