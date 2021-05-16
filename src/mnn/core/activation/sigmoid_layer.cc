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

std::string SigmoidLayer::layer_type() const
{
    return "sigmoid-activation";
}

void SigmoidLayer::forward_activation(const Vector &x, Vector &y)
{
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = Float(1) / (Float(1) + std::exp(-x[j]));
    }
}

void SigmoidLayer::backward_activation(const Vector &x, const Vector &y,
        Vector &dx, const Vector &dy)
{
    for (size_t j = 0; j < x.size(); j++) {
        // dx = dy * (gradient of sigmoid)
        dx[j] = dy[j] * y[j] * (Float(1) - y[j]);
    }
}

std::pair<Float, Float> SigmoidLayer::scale() const
{
    return std::make_pair(Float(0.1), Float(0.9));
}
}  // namespace mnn
