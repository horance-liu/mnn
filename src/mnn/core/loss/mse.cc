/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/loss/mse.h"

namespace mnn {

Float Mse::f(const Vector &y, const Vector &t)
{
    assert(y.size() == t.size());
    Float d { 0.0 };

    for (size_t i = 0; i < y.size(); ++i)
        d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<Float>(y.size());
}

Vector Mse::df(const Vector &y, const Vector &t)
{
    assert(y.size() == t.size());
    Vector d(t.size());
    Float factor = Float(2) / static_cast<Float>(t.size());

    for (size_t i = 0; i < y.size(); ++i)
        d[i] = factor * (y[i] - t[i]);

    return d;
}

}  // namespace mnn
