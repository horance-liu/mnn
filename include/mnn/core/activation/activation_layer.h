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

class activation_layer: public layer {
public:
    activation_layer();
    activation_layer(size_t in_width, size_t in_height, size_t in_channels);

    explicit activation_layer(size_t in_dim);
    explicit activation_layer(const shape3d &in_shape);
    explicit activation_layer(const layer &prev_layer);

private:
    std::vector<shape3d> in_shape() const override;
    std::vector<shape3d> out_shape() const override;

    void set_in_shape(const shape3d &in_shape) override;

    void forward_propagation(
            const std::vector<tensor_t*> &in_data,
            std::vector<tensor_t*> &out_data) override;

    void back_propagation(
            const std::vector<tensor_t*> &in_data,
            const std::vector<tensor_t*> &out_data,
            std::vector<tensor_t*> &out_grad,
            std::vector<tensor_t*> &in_grad) override;

private:
    std::string layer_type() const override = 0;

    virtual void forward_activation(const vec_t &x, vec_t &y) = 0;

    virtual void backward_activation(
            const vec_t &x,
            const vec_t &y,
            vec_t &dx,
            const vec_t &dy) = 0;

    virtual std::pair<float_t, float_t> scale() const = 0;

private:
    shape3d in_shape_;
};

}  // namespace mnn
