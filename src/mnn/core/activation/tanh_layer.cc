/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/activation/tanh_layer.h"

#include <string>
#include <utility>

namespace mnn {

std::string TanhLayer::layer_type() const
{
    return "tanh-activation";
}

void TanhLayer::forward_activation(const Vector &x, Vector &y)
{
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = std::tanh(x[j]);
    }
}

void TanhLayer::backward_activation(const Vector &x, const Vector &y, Vector &dx,
        const Vector &dy)
{
    for (size_t j = 0; j < x.size(); j++) {
        // dx = dy * (gradient of tanh)
        dx[j] = dy[j] * (Float(1) - sqr(y[j]));
    }
}

std::pair<Float, Float> TanhLayer::scale() const
{
    return std::make_pair(Float(-0.8), Float(0.8));
}

}  // namespace mnn
