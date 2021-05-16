/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/core/loss/cross_entropy.h"

namespace mnn {

float_t cross_entropy::f(const vec_t &y, const vec_t &t)
{
    assert(y.size() == t.size());
    float_t d { 0 };

    for (size_t i = 0; i < y.size(); ++i)
        d += -t[i] * std::log(y[i])
                - (float_t(1) - t[i]) * std::log(float_t(1) - y[i]);

    return d;
}

vec_t cross_entropy::df(const vec_t &y, const vec_t &t)
{
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i)
        d[i] = (y[i] - t[i]) / (y[i] * (float_t(1) - y[i]));

    return d;
}

}  // namespace mnn
