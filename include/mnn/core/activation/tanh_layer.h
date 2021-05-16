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

class TanhLayer: public ActivationLayer {
public:
    using ActivationLayer::ActivationLayer;

private:
    std::string layer_type() const override;
    void forward_activation(const Vector &x, Vector &y) override;
    void backward_activation(const Vector &x, const Vector &y, Vector &dx, const Vector &dy) override;
    std::pair<Float, Float> scale() const override;
};

}  // namespace mnn
