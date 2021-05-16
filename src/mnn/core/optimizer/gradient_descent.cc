/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/optimizer/gradient_descent.h"

namespace mnn {

gradient_descent::gradient_descent() : alpha(float_t(0.01)), lambda(float_t(0))
{
}

void gradient_descent::update(const vec_t &dW, vec_t &W, bool parallelize)
{
    for_i(parallelize, W.size(),
            [&](size_t i) {W[i] = W[i] - alpha * (dW[i] + lambda * W[i]);});
}

}  // namespace mnn
