/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/activation/activation_layer.h"
#include <string>
#include <utility>
#include <vector>

namespace mnn {

ActivationLayer::ActivationLayer() : ActivationLayer(Shape3d(0, 0, 0))
{
}

ActivationLayer::ActivationLayer(size_t in_dim)
    : ActivationLayer(Shape3d(in_dim, 1, 1))
{
}

ActivationLayer::ActivationLayer(
        size_t in_width,
        size_t in_height,
        size_t in_channels)
   : ActivationLayer(Shape3d(in_width, in_height, in_channels))
{
}

ActivationLayer::ActivationLayer(const Shape3d &in_shape)
    : Layer( {VectorType::DATA }, { VectorType::DATA }), in_shape_(in_shape)
{
}

ActivationLayer::ActivationLayer(const Layer &prev_layer) :
        Layer( {VectorType::DATA }, { VectorType::DATA }),
        in_shape_(prev_layer.out_shape()[0])
{
}

std::vector<Shape3d> ActivationLayer::in_shape() const
{
    return {in_shape_};
}

std::vector<Shape3d> ActivationLayer::out_shape() const
{
    return {in_shape_};
}

void ActivationLayer::set_in_shape(const Shape3d &in_shape)
{
    this->in_shape_ = in_shape;
}

void ActivationLayer::forward_propagation(
        const std::vector<Matrix*> &in_data,
        std::vector<Matrix*> &out_data)
{
    const Matrix &x = *in_data[0];
    Matrix &y = *out_data[0];

    for_i(x.size(), [&](size_t i) {
        forward_activation(x[i], y[i]);
    });
}

void ActivationLayer::back_propagation(
        const std::vector<Matrix*> &in_data,
        const std::vector<Matrix*> &out_data,
        std::vector<Matrix*> &out_grad,
        std::vector<Matrix*> &in_grad)
{
    Matrix &dx = *in_grad[0];
    const Matrix &dy = *out_grad[0];
    const Matrix &x = *in_data[0];
    const Matrix &y = *out_data[0];

    for_i(x.size(), [&](size_t i) {
        backward_activation(x[i], y[i], dx[i], dy[i]);
    });
}

} // namespace mnn
