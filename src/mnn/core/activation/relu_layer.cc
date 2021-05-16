/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/activation/relu_layer.h"
#include <algorithm>
#include <string>
#include <utility>

namespace mnn {

std::string ReluLayer::layer_type() const
{
    return "relu-activation";
}

void ReluLayer::forward_activation(const Vector &x, Vector &y)
{
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = std::max(Float(0), x[j]);
    }
}

void ReluLayer::backward_activation(
        const Vector &x,
        const Vector &y,
        Vector &dx,
        const Vector &dy)
{
    for (size_t j = 0; j < x.size(); j++) {
        // dx = dy * (gradient of relu)
        dx[j] = dy[j] * (y[j] > Float(0) ? Float(1) : Float(0));
    }
}

std::pair<Float, Float> ReluLayer::scale() const
{
    return std::make_pair(Float(0.1), Float(0.9));
}

}  // namespace mnn
