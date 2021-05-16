/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/layer/layer.h"
#include "mnn/infra/util.h"

namespace mnn {

class ActivationLayer: public Layer {
public:
    ActivationLayer();
    ActivationLayer(size_t in_width, size_t in_height, size_t in_channels);

    explicit ActivationLayer(size_t in_dim);
    explicit ActivationLayer(const Shape3d &in_shape);
    explicit ActivationLayer(const Layer &prev_layer);

private:
    std::vector<Shape3d> in_shape() const override;
    std::vector<Shape3d> out_shape() const override;

    void set_in_shape(const Shape3d &in_shape) override;

    void forward_propagation(
            const std::vector<Matrix*> &in_data,
            std::vector<Matrix*> &out_data) override;

    void back_propagation(
            const std::vector<Matrix*> &in_data,
            const std::vector<Matrix*> &out_data,
            std::vector<Matrix*> &out_grad,
            std::vector<Matrix*> &in_grad) override;

private:
    std::string layer_type() const override = 0;

    virtual void forward_activation(const Vector &x, Vector &y) = 0;

    virtual void backward_activation(
            const Vector &x,
            const Vector &y,
            Vector &dx,
            const Vector &dy) = 0;

    virtual std::pair<Float, Float> scale() const = 0;

private:
    Shape3d in_shape_;
};

}  // namespace mnn
