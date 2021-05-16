/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/core/layer/fully_connected_layer.h"
#include "mnn/op/fully_connected_grad_op.h"
#include "mnn/op/fully_connected_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mnn {

fully_connected_layer::fully_connected_layer(size_t in_dim, size_t out_dim,
        bool has_bias, backend_t backend_type) : layer(
        std_input_order(has_bias), { vector_type::data })
{
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
}

fully_connected_layer::fully_connected_layer(fully_connected_layer &&other) : layer(
        std::move(other)), params_(std::move(other.params_)), kernel_fwd_(
        std::move(other.kernel_fwd_)), kernel_back_(
        std::move(other.kernel_back_))
{
    init_backend(std::move(other.engine()));
}

size_t fully_connected_layer::fan_in_size() const
{
    return params_.in_size_;
}
size_t fully_connected_layer::fan_out_size() const
{
    return params_.out_size_;
}

std::vector<index3d<size_t>> fully_connected_layer::in_shape() const
{
    if (params_.has_bias_) {
        return {index3d<size_t>(params_.in_size_, 1, 1),
            index3d<size_t>(params_.in_size_, params_.out_size_, 1),
            index3d<size_t>(params_.out_size_, 1, 1)};
    } else {
        return {index3d<size_t>(params_.in_size_, 1, 1),
            index3d<size_t>(params_.in_size_, params_.out_size_, 1)};
    }
}

std::vector<index3d<size_t>> fully_connected_layer::out_shape() const
{
    return {index3d<size_t>(params_.out_size_, 1, 1)};
}

void fully_connected_layer::forward_propagation(
        const std::vector<tensor_t*> &in_data, std::vector<tensor_t*> &out_data)
{
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setEngine(layer::engine());

    kernel_fwd_->compute(fwd_ctx_);
}

void fully_connected_layer::back_propagation(
        const std::vector<tensor_t*> &in_data,
        const std::vector<tensor_t*> &out_data,
        std::vector<tensor_t*> &out_grad, std::vector<tensor_t*> &in_grad)
{
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setEngine(layer::engine());

    kernel_back_->compute(bwd_ctx_);
}

std::string fully_connected_layer::layer_type() const
{
    return "fully-connected";
}

void fully_connected_layer::set_params(const size_t in_size,
        const size_t out_size, bool has_bias)
{
    params_.in_size_ = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
}

void fully_connected_layer::init_backend(backend_t backend_type)
{
    OpKernelConstruction ctx = OpKernelConstruction(&params_);

    if (backend_type == backend_t::cpu) {
        kernel_fwd_.reset(new FullyConnectedOp(ctx));
        kernel_back_.reset(new FullyConnectedGradOp(ctx));
    } else {
        throw nn_error("Not supported engine: " + to_string(backend_type));
    }
}

}  // namespace mnn
