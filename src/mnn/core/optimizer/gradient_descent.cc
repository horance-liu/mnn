/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/optimizer/gradient_descent.h"

namespace mnn {

GradientDescent::GradientDescent() : alpha(Float(0.01)), lambda(Float(0))
{
}

void GradientDescent::update(const Vector &dW, Vector &W, bool parallelize)
{
    for_i(parallelize, W.size(),
            [&](size_t i) {W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);});
}

}  // namespace mnn
