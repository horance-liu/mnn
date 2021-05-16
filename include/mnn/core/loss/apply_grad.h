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
vec_t gradient(const vec_t &y, const vec_t &t)
{
    assert(y.size() == t.size());
    return E::df(y, t);
}

template<typename E>
std::vector<vec_t> gradient(
        const std::vector<vec_t> &y,
        const std::vector<vec_t> &t)
{
    std::vector<vec_t> grads(y.size());

    assert(y.size() == t.size());

    for (size_t i = 0; i < y.size(); i++)
        grads[i] = gradient<E>(y[i], t[i]);

    return grads;
}

void apply_cost_if_defined(
        std::vector<vec_t> &sample_gradient,
        const std::vector<vec_t> &sample_cost);

template<typename E>
std::vector<tensor_t> gradient(const std::vector<tensor_t> &y,
        const std::vector<tensor_t> &t, const std::vector<tensor_t> &t_cost)
{
    const size_t sample_count = y.size();

    std::vector<tensor_t> gradients(sample_count);

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
