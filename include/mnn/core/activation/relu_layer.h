/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/activation/activation_layer.h"

namespace mnn {

class relu_layer: public activation_layer {
public:
    using activation_layer::activation_layer;

private:
    std::string layer_type() const override;
    void forward_activation(const vec_t &x, vec_t &y) override;
    void backward_activation(const vec_t &x, const vec_t &y, vec_t &dx, const vec_t &dy) override;
    std::pair<float_t, float_t> scale() const override;
};

}  // namespace mnn
