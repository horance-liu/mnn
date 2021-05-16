/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/optimizer/adam.h"

namespace mnn {

adam::adam() : alpha(float_t(0.001)), b1(float_t(0.9)), b2(float_t(0.999)), b1_t(
        float_t(0.9)), b2_t(float_t(0.999)), eps(float_t(1e-8))
{
}

void adam::update(const vec_t &dW, vec_t &W, bool parallelize)
{
    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
        mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
        vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

        // L2 norm based update rule
            W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
            std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
        });

    b1_t *= b1;
    b2_t *= b2;
}

}  // namespace mnn
