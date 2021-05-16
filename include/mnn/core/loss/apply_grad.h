/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/util.h"

namespace mnn {

template<typename E>
Vector gradient(const Vector &y, const Vector &t)
{
    assert(y.size() == t.size());
    return E::df(y, t);
}

template<typename E>
std::vector<Vector> gradient(
        const std::vector<Vector> &y,
        const std::vector<Vector> &t)
{
    std::vector<Vector> grads(y.size());

    assert(y.size() == t.size());

    for (size_t i = 0; i < y.size(); i++)
        grads[i] = gradient<E>(y[i], t[i]);

    return grads;
}

void apply_cost_if_defined(
        std::vector<Vector> &sample_gradient,
        const std::vector<Vector> &sample_cost);

template<typename E>
std::vector<Matrix> gradient(const std::vector<Matrix> &y,
        const std::vector<Matrix> &t, const std::vector<Matrix> &t_cost)
{
    const size_t sample_count = y.size();

    std::vector<Matrix> gradients(sample_count);

    assert(y.size() == t.size());
    assert(t_cost.empty() || t_cost.size() == t.size());

    for (size_t sample = 0; sample < sample_count; ++sample) {
        gradients[sample] = gradient<E>(y[sample], t[sample]);

        if (sample < t_cost.size()) {
            apply_cost_if_defined(gradients[sample], t_cost[sample]);
        }
    }

    return gradients;
}

}  // namespace mnn
