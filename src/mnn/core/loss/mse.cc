/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/loss/mse.h"

namespace mnn {

float_t mse::f(const vec_t &y, const vec_t &t)
{
    assert(y.size() == t.size());
    float_t d { 0.0 };

    for (size_t i = 0; i < y.size(); ++i)
        d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<float_t>(y.size());
}

vec_t mse::df(const vec_t &y, const vec_t &t)
{
    assert(y.size() == t.size());
    vec_t d(t.size());
    float_t factor = float_t(2) / static_cast<float_t>(t.size());

    for (size_t i = 0; i < y.size(); ++i)
        d[i] = factor * (y[i] - t[i]);

    return d;
}

}  // namespace mnn
