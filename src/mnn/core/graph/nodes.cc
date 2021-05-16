/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/nodes.h"

#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mnn/core/layer/layer.h"
#include "mnn/core/optimizer/optimizer.h"
#include "mnn/infra/util.h"

namespace mnn {

void nodes::update_weights(optimizer *opt)
{
    for (auto l : nodes_) {
        l->update_weight(opt);
    }
}

void nodes::setup(bool reset_weight)
{
    for (auto l : nodes_) {
        l->setup(reset_weight);
    }
}

void nodes::clear_grads()
{
    for (auto l : nodes_) {
        l->clear_grads();
    }
}

size_t nodes::size() const
{
    return nodes_.size();
}

nodes::iterator nodes::begin()
{
    return nodes_.begin();
}
nodes::iterator nodes::end()
{
    return nodes_.end();
}

nodes::const_iterator nodes::begin() const
{
    return nodes_.begin();
}

nodes::const_iterator nodes::end() const
{
    return nodes_.end();
}

size_t nodes::in_data_size() const
{
    return nodes_.front()->in_data_size();
}
size_t nodes::out_data_size() const
{
    return nodes_.back()->out_data_size();
}

// @todo: multiple output
float_t nodes::target_value_min(int out_channel) const
{
    MNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().first;
}

float_t nodes::target_value_max(int out_channel) const
{
    MNN_UNREFERENCED_PARAMETER(out_channel);
    return nodes_.back()->out_value_range().second;
}

void nodes::label2vec(const label_t *t, size_t num,
        std::vector<vec_t> &vec) const
{
    size_t outdim = out_data_size();

    vec.reserve(num);
    for (size_t i = 0; i < num; i++) {
        assert(t[i] < outdim);
        vec.emplace_back(outdim, target_value_min());
        vec.back()[t[i]] = target_value_max();
    }
}

void nodes::label2vec(const std::vector<label_t> &labels,
        std::vector<vec_t> &vec) const
{
    return label2vec(&labels[0], labels.size(), vec);
}

// transform indexing so that it's more suitable for per-layer operations
// input:  [sample][channel][feature]
// output: [channel][sample][feature]
void nodes::reorder_for_layerwise_processing(const std::vector<tensor_t> &input,
        std::vector<std::vector<const vec_t*>> &output)
{
    size_t sample_count = input.size();
    size_t channel_count = input[0].size();

    output.resize(channel_count);
    for (size_t i = 0; i < channel_count; ++i) {
        output[i].resize(sample_count);
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        assert(input[sample].size() == channel_count);
        for (size_t channel = 0; channel < channel_count; ++channel) {
            output[channel][sample] = &input[sample][channel];
        }
    }
}
}  // namespace mnn
