/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/activation/sigmoid_layer.h"

#include <string>
#include <utility>

namespace mnn {

std::string sigmoid_layer::layer_type() const
{
    return "sigmoid-activation";
}

void sigmoid_layer::forward_activation(const vec_t &x, vec_t &y)
{
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = float_t(1) / (float_t(1) + std::exp(-x[j]));
    }
}

void sigmoid_layer::backward_activation(const vec_t &x, const vec_t &y,
        vec_t &dx, const vec_t &dy)
{
    for (size_t j = 0; j < x.size(); j++) {
        // dx = dy * (gradient of sigmoid)
        dx[j] = dy[j] * y[j] * (float_t(1) - y[j]);
    }
}

std::pair<float_t, float_t> sigmoid_layer::scale() const
{
    return std::make_pair(float_t(0.1), float_t(0.9));
}
}  // namespace mnn
