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

std::string tanh_layer::layer_type() const
{
    return "tanh-activation";
}

void tanh_layer::forward_activation(const vec_t &x, vec_t &y)
{
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = std::tanh(x[j]);
    }
}

void tanh_layer::backward_activation(const vec_t &x, const vec_t &y, vec_t &dx,
        const vec_t &dy)
{
    for (size_t j = 0; j < x.size(); j++) {
        // dx = dy * (gradient of tanh)
        dx[j] = dy[j] * (float_t(1) - sqr(y[j]));
    }
}

std::pair<float_t, float_t> tanh_layer::scale() const
{
    return std::make_pair(float_t(-0.8), float_t(0.8));
}

}  // namespace mnn
