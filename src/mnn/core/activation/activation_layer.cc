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

activation_layer::activation_layer() : activation_layer(shape3d(0, 0, 0))
{
}

activation_layer::activation_layer(size_t in_dim)
    : activation_layer(shape3d(in_dim, 1, 1))
{
}

activation_layer::activation_layer(
        size_t in_width,
        size_t in_height,
        size_t in_channels)
   : activation_layer(shape3d(in_width, in_height, in_channels))
{
}

activation_layer::activation_layer(const shape3d &in_shape)
    : layer( {vector_type::data }, { vector_type::data }), in_shape_(in_shape)
{
}

activation_layer::activation_layer(const layer &prev_layer) :
        layer( {vector_type::data }, { vector_type::data }),
        in_shape_(prev_layer.out_shape()[0])
{
}

std::vector<shape3d> activation_layer::in_shape() const
{
    return {in_shape_};
}

std::vector<shape3d> activation_layer::out_shape() const
{
    return {in_shape_};
}

void activation_layer::set_in_shape(const shape3d &in_shape)
{
    this->in_shape_ = in_shape;
}

void activation_layer::forward_propagation(
        const std::vector<tensor_t*> &in_data,
        std::vector<tensor_t*> &out_data)
{
    const tensor_t &x = *in_data[0];
    tensor_t &y = *out_data[0];

    for_i(x.size(), [&](size_t i) {
        forward_activation(x[i], y[i]);
    });
}

void activation_layer::back_propagation(
        const std::vector<tensor_t*> &in_data,
        const std::vector<tensor_t*> &out_data,
        std::vector<tensor_t*> &out_grad,
        std::vector<tensor_t*> &in_grad)
{
    tensor_t &dx = *in_grad[0];
    const tensor_t &dy = *out_grad[0];
    const tensor_t &x = *in_data[0];
    const tensor_t &y = *out_data[0];

    for_i(x.size(), [&](size_t i) {
        backward_activation(x[i], y[i], dx[i], dy[i]);
    });
}

} // namespace mnn
