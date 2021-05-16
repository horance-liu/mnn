/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/activation/softmax_layer.h"

#include <string>
#include <utility>

namespace mnn {

std::string SoftmaxLayer::layer_type() const
{
    return "softmax-activation";
}

void SoftmaxLayer::forward_activation(const Vector &x, Vector &y)
{
    const Float alpha = *std::max_element(x.begin(), x.end());
    Float denominator(0);
    for (size_t j = 0; j < x.size(); j++) {
        y[j] = std::exp(x[j] - alpha);
        denominator += y[j];
    }
    for (size_t j = 0; j < x.size(); j++) {
        y[j] /= denominator;
    }
}

void SoftmaxLayer::backward_activation(const Vector &x, const Vector &y,
        Vector &dx, const Vector &dy)
{
    const size_t len = dy.size();

#if HAS_CXX11_THREAD_LOCAL
    thread_local
#endif
    Vector df(len, 0);
    for (size_t j = 0; j < x.size(); j++) {
        for (size_t k = 0; k < x.size(); k++) {
            df[k] = (k == j) ? y[j] * (Float(1) - y[j]) : -y[k] * y[j];
        }
        // dx = dy * (gradient of softmax)
        dx[j] = vectorize::dot(&dy[0], &df[0], len);
    }
}

std::pair<Float, Float> SoftmaxLayer::scale() const
{
    return std::make_pair(Float(0), Float(1));
}

}  // namespace mnn
