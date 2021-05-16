/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/layer/convolutional_layer.h"

#include "mnn/op/conv2d_grad_op.h"
#include "mnn/op/conv2d_op.h"
#include "mnn/infra/util.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>


namespace mnn {

convolutional_layer::convolutional_layer(size_t in_width, size_t in_height,
        size_t window_size, size_t in_channels, size_t out_channels,
        padding pad_type, bool has_bias, size_t w_stride, size_t h_stride,
        size_t w_dilation, size_t h_dilation, backend_t backend_type) : convolutional_layer(
        in_width, in_height, window_size, window_size, in_channels,
        out_channels, connection_table(), pad_type, has_bias, w_stride,
        h_stride, w_dilation, h_dilation, backend_type)
{
}

convolutional_layer::convolutional_layer(size_t in_width, size_t in_height,
        size_t window_width, size_t window_height, size_t in_channels,
        size_t out_channels, padding pad_type, bool has_bias, size_t w_stride,
        size_t h_stride, size_t w_dilation, size_t h_dilation,
        backend_t backend_type) : convolutional_layer(in_width, in_height,
        window_width, window_height, in_channels, out_channels,
        connection_table(), pad_type, has_bias, w_stride, h_stride, w_dilation,
        h_dilation, backend_type)
{
}

convolutional_layer::convolutional_layer(size_t in_width, size_t in_height,
        size_t window_size, size_t in_channels, size_t out_channels,
        const connection_table &connection_table, padding pad_type,
        bool has_bias, size_t w_stride, size_t h_stride, size_t w_dilation,
        size_t h_dilation, backend_t backend_type) : convolutional_layer(
        in_width, in_height, window_size, window_size, in_channels,
        out_channels, connection_table, pad_type, has_bias, w_stride, h_stride,
        w_dilation, h_dilation, backend_type)
{
}

convolutional_layer::convolutional_layer(size_t in_width, size_t in_height,
        size_t window_width, size_t window_height, size_t in_channels,
        size_t out_channels, const connection_table &connection_table,
        padding pad_type, bool has_bias, size_t w_stride, size_t h_stride,
        size_t w_dilation, size_t h_dilation, backend_t backend_type) : layer(
        std_input_order(has_bias), { vector_type::data })
{
    conv_set_params(shape3d(in_width, in_height, in_channels), window_width,
            window_height, out_channels, pad_type, has_bias, w_stride, h_stride,
            w_dilation, h_dilation, connection_table);
    init_backend(backend_type);
    layer::set_backend_type(backend_type);
}

convolutional_layer::convolutional_layer(convolutional_layer &&other)  // NOLINT
: layer(std::move(other)), params_(std::move(other.params_)), padding_op_(
        std::move(other.padding_op_)), kernel_fwd_(
        std::move(other.kernel_fwd_)), kernel_back_(
        std::move(other.kernel_back_)), cws_(std::move(other.cws_))
{
    init_backend(std::move(other.engine()));
}

size_t convolutional_layer::fan_in_size() const
{
    return params_.weight.width_ * params_.weight.height_ * params_.in.depth_;
}

size_t convolutional_layer::fan_out_size() const
{
    return (params_.weight.width_ / params_.w_stride)
            * (params_.weight.height_ / params_.h_stride) * params_.out.depth_;
}

void convolutional_layer::forward_propagation(
        const std::vector<tensor_t*> &in_data, std::vector<tensor_t*> &out_data)
{
    padding_op_.copy_and_pad_input(*in_data[0], cws_.prev_out_padded_);

    fwd_in_data_.resize(in_data.size());
    std::copy(in_data.begin(), in_data.end(), fwd_in_data_.begin());
    fwd_in_data_[0] = in_data_padded(in_data);

    fwd_ctx_.set_in_out(fwd_in_data_, out_data);
    fwd_ctx_.setParallelize(layer::parallelize());
    fwd_ctx_.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_fwd_->compute(fwd_ctx_);
}

void convolutional_layer::back_propagation(
        const std::vector<tensor_t*> &in_data,
        const std::vector<tensor_t*> &out_data,
        std::vector<tensor_t*> &out_grad, std::vector<tensor_t*> &in_grad)
{
    bwd_in_data_.resize(in_data.size());
    std::copy(in_data.begin(), in_data.end(), bwd_in_data_.begin());
    bwd_in_data_[0] = in_data_padded(in_data);

    bwd_in_grad_.resize(in_grad.size());
    std::copy(in_grad.begin(), in_grad.end(), bwd_in_grad_.begin());
    if (params_.pad_type == padding::same) {
        bwd_in_grad_[0] = &cws_.prev_delta_padded_;
    }

    bwd_ctx_.set_in_out(bwd_in_data_, out_data, out_grad, bwd_in_grad_);
    bwd_ctx_.setParams(&params_);
    bwd_ctx_.setParallelize(layer::parallelize());
    bwd_ctx_.setEngine(layer::engine());

    // launch convolutional kernel
    kernel_back_->compute(bwd_ctx_);

    // unpad deltas
    padding_op_.copy_and_unpad_delta(cws_.prev_delta_padded_, *in_grad[0]);
}

void convolutional_layer::set_sample_count(size_t sample_count)
{
    layer::set_sample_count(sample_count);
    cws_.prev_delta_padded_.resize(sample_count,
            vec_t(params_.in_padded.size(), float_t(0)));
}

std::vector<index3d<size_t>> convolutional_layer::in_shape() const
{
    if (params_.has_bias) {
        return {params_.in, params_.weight,
            index3d<size_t>(1, 1, params_.out.depth_)};
    } else {
        return {params_.in, params_.weight};
    }
}

std::vector<index3d<size_t>> convolutional_layer::out_shape() const
{
    return {params_.out};
}

std::string convolutional_layer::layer_type() const
{
    return std::string("conv");
}

tensor_t* convolutional_layer::in_data_padded(const std::vector<tensor_t*> &in)
{
    return (params_.pad_type == padding::valid) ? in[0] : &cws_.prev_out_padded_;
}

void convolutional_layer::conv_set_params(const shape3d &in, size_t w_width,
        size_t w_height, size_t outc, padding ptype, bool has_bias,
        size_t w_stride, size_t h_stride, size_t w_dilation, size_t h_dilation,
        const connection_table &tbl)
{
    params_.in = in;
    params_.in_padded = shape3d(in_length(in.width_, w_width, ptype),
            in_length(in.height_, w_height, ptype), in.depth_);
    params_.out = shape3d(
            conv_out_length(in.width_, w_width, w_stride, w_dilation, ptype),
            conv_out_length(in.height_, w_height, h_stride, h_dilation, ptype),
            outc);
    params_.weight = shape3d(w_width, w_height, in.depth_ * outc);
    params_.has_bias = has_bias;
    params_.pad_type = ptype;
    params_.w_stride = w_stride;
    params_.h_stride = h_stride;
    params_.w_dilation = w_dilation;
    params_.h_dilation = h_dilation;
    params_.tbl = tbl;

    // init padding buffer
    if (params_.pad_type == padding::same) {
        cws_.prev_delta_padded_.resize(1,
                vec_t(params_.in_padded.size(), float_t(0)));
    }

    // set parameters to padding operation
    padding_op_ = Conv2dPadding(params_);
}

size_t convolutional_layer::in_length(size_t in_length, size_t window_size,
        padding pad_type) const
{
    return pad_type == padding::same ? (in_length + window_size - 1) : in_length;
}

size_t convolutional_layer::conv_out_dim(size_t in_width,
        size_t in_height, size_t window_size, size_t w_stride, size_t h_stride,
        size_t w_dilation, size_t h_dilation, padding pad_type)
{
    return conv_out_length(in_width, window_size, w_stride, w_dilation,
            pad_type)
            * conv_out_length(in_height, window_size, h_stride, h_dilation,
                    pad_type);
}

size_t convolutional_layer::conv_out_dim(size_t in_width, size_t in_height,
        size_t window_width, size_t window_height, size_t w_stride,
        size_t h_stride, size_t w_dilation, size_t h_dilation,
        padding pad_type) const
{
    return conv_out_length(in_width, window_width, w_stride, w_dilation,
            pad_type)
            * conv_out_length(in_height, window_height, h_stride, h_dilation,
                    pad_type);
}

void convolutional_layer::init_backend(const backend_t backend_type)
{
    OpKernelConstruction ctx = OpKernelConstruction(&params_);

    if (backend_type == backend_t::cpu) {
        kernel_fwd_.reset(new Conv2dOp(ctx));
        kernel_back_.reset(new Conv2dGradOp(ctx));
        return;
    } else {
        throw nn_error("Not supported engine: " + to_string(backend_type));
    }
}

}  // namespace mnn
