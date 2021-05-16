/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 * 
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/graph/sequential.h"

namespace mnn {

void Sequential::backward(const std::vector<Matrix> &first)
{
    std::vector<std::vector<const Vector*>> reordered_grad;
    reorder_for_layerwise_processing(first, reordered_grad);
    assert(reordered_grad.size() == 1);

    nodes_.back()->set_out_grads(&reordered_grad[0], 1);

    for (auto l = nodes_.rbegin(); l != nodes_.rend(); l++) {
        (*l)->backward();
    }
}

std::vector<Matrix> Sequential::forward(const std::vector<Matrix> &first)
{
    std::vector<std::vector<const Vector*>> reordered_data;
    reorder_for_layerwise_processing(first, reordered_data);
    assert(reordered_data.size() == 1);

    nodes_.front()->set_in_data(&reordered_data[0], 1);

    for (auto l : nodes_) {
        l->forward();
    }

    std::vector<const Matrix*> out;
    nodes_.back()->output(out);

    return normalize_out(out);
}

void Sequential::check_connectivity()
{
    for (size_t i = 0; i < nodes_.size() - 1; i++) {
        auto out = nodes_[i]->outputs();
        auto in = nodes_[i + 1]->inputs();

        if (out[0] != in[0]) {
            throw MnnError("");
        }
    }
}

std::vector<Matrix> Sequential::normalize_out(
        const std::vector<const Matrix*> &out)
{
    std::vector < Matrix > normalized_output;

    const size_t sample_count = out[0]->size();
    normalized_output.resize(sample_count, Matrix(1));

    for (size_t sample = 0; sample < sample_count; ++sample) {
        normalized_output[sample][0] = (*out[0])[sample];
    }

    return normalized_output;
}

} // namespace mnn
