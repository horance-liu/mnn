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

FullyConnectedLayer::FullyConnectedLayer(size_t in_dim, size_t out_dim,
        bool has_bias, BackendType backend_type) : Layer(
        std_input_order(has_bias), { VectorType::DATA })
{
    set_params(in_dim, out_dim, has_bias);
    init_backend(backend_type);
    Layer::set_backend_type(backend_type);
}

FullyConnectedLayer::FullyConnectedLayer(FullyConnectedLayer &&other) : Layer(
        std::move(other)), params_(std::move(other.params_)), kernel_fwd_(
        std::move(other.kernel_fwd_)), kernel_back_(
        std::move(other.kernel_back_))
{
    init_backend(std::move(other.engine()));
}

size_t FullyConnectedLayer::fan_in_size() const
{
    return params_.in_size_;
}
size_t FullyConnectedLayer::fan_out_size() const
{
    return params_.out_size_;
}

std::vector<Shape3d> FullyConnectedLayer::in_shape() const
{
    if (params_.has_bias_) {
        return {Shape3d(params_.in_size_, 1, 1),
            Shape3d(params_.in_size_, params_.out_size_, 1),
            Shape3d(params_.out_size_, 1, 1)};
    } else {
        return {Shape3d(params_.in_size_, 1, 1),
            Shape3d(params_.in_size_, params_.out_size_, 1)};
    }
}

std::vector<Shape3d> FullyConnectedLayer::out_shape() const
{
    return {Shape3d(params_.out_size_, 1, 1)};
}

void FullyConnectedLayer::forward_propagation(
        const std::vector<Matrix*> &in_data, std::vector<Matrix*> &out_data)
{
    fwd_ctx_.set_in_out(in_data, out_data);
    fwd_ctx_.setParallelize(Layer::parallelize());
    fwd_ctx_.setEngine(Layer::engine());

    kernel_fwd_->compute(fwd_ctx_);
}

void FullyConnectedLayer::back_propagation(
        const std::vector<Matrix*> &in_data,
        const std::vector<Matrix*> &out_data,
        std::vector<Matrix*> &out_grad, std::vector<Matrix*> &in_grad)
{
    bwd_ctx_.set_in_out(in_data, out_data, out_grad, in_grad);
    bwd_ctx_.setParallelize(Layer::parallelize());
    bwd_ctx_.setEngine(Layer::engine());

    kernel_back_->compute(bwd_ctx_);
}

std::string FullyConnectedLayer::layer_type() const
{
    return "fully-connected";
}

void FullyConnectedLayer::set_params(const size_t in_size,
        const size_t out_size, bool has_bias)
{
    params_.in_size_ = in_size;
    params_.out_size_ = out_size;
    params_.has_bias_ = has_bias;
}

void FullyConnectedLayer::init_backend(BackendType backend_type)
{
    OpKernelConstruction ctx = OpKernelConstruction(&params_);

    if (backend_type == BackendType::CPU) {
        kernel_fwd_.reset(new FullyConnectedOp(ctx));
        kernel_back_.reset(new FullyConnectedGradOp(ctx));
    } else {
        throw MnnError("Not supported engine: " + to_string(backend_type));
    }
}

}  // namespace mnn
