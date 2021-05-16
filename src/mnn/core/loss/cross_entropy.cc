/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/core/loss/cross_entropy.h"

namespace mnn {

Float CrossEntropy::f(const Vector &y, const Vector &t)
{
    assert(y.size() == t.size());
    Float d { 0 };

    for (size_t i = 0; i < y.size(); ++i)
        d += -t[i] * std::log(y[i])
                - (Float(1) - t[i]) * std::log(Float(1) - y[i]);

    return d;
}

Vector CrossEntropy::df(const Vector &y, const Vector &t)
{
    assert(y.size() == t.size());
    Vector d(t.size());

    for (size_t i = 0; i < y.size(); ++i)
        d[i] = (y[i] - t[i]) / (y[i] * (Float(1) - y[i]));

    return d;
}

}  // namespace mnn
