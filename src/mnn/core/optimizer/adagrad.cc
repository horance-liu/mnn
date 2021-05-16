/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/optimizer/adagrad.h"

namespace mnn {

Adagrad::Adagrad() : alpha(Float(0.01)), eps(Float(1e-8))
{
}

void Adagrad::update(const Vector &dW, Vector &W, bool parallelize)
{
    Vector &g = get<0>(W);
    for_i(parallelize, W.size(), [&](size_t i) {
        g[i] += dW[i] * dW[i];
        W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
    });
}

}  // namespace mnn
