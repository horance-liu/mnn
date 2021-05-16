/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/op_kernel.h"
#include "mnn/infra/backend.h"
#include "mnn/core/layer/layer.h"

namespace mnn {

class convolutional_layer: public layer {
public:
    convolutional_layer(size_t in_width, size_t in_height, size_t window_size,
            size_t in_channels, size_t out_channels, padding pad_type =
                    padding::valid, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            backend_t backend_type = default_engine());

    convolutional_layer(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t in_channels, size_t out_channels,
            padding pad_type = padding::valid, bool has_bias = true,
            size_t w_stride = 1, size_t h_stride = 1, size_t w_dilation = 1,
            size_t h_dilation = 1, backend_t backend_type = default_engine());

    convolutional_layer(size_t in_width, size_t in_height, size_t window_size,
            size_t in_channels, size_t out_channels,
            const connection_table &connection_table, padding pad_type =
                    padding::valid, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            backend_t backend_type = default_engine());

    convolutional_layer(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t in_channels, size_t out_channels,
            const connection_table &connection_table, padding pad_type =
                    padding::valid, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            backend_t backend_type = default_engine());

    convolutional_layer(convolutional_layer &&other);

    size_t fan_in_size() const override;
    size_t fan_out_size() const override;

    void forward_propagation(
            const std::vector<tensor_t*> &in_data,
            std::vector<tensor_t*> &out_data) override;

    void back_propagation(
            const std::vector<tensor_t*> &in_data,
            const std::vector<tensor_t*> &out_data,
            std::vector<tensor_t*> &out_grad,
            std::vector<tensor_t*> &in_grad) override;

    void set_sample_count(size_t sample_count) override;
    std::string layer_type() const override;

    std::vector<index3d<size_t>> in_shape() const override;
    std::vector<index3d<size_t>> out_shape() const override;

private:
    tensor_t* in_data_padded(const std::vector<tensor_t*> &in);
    void conv_set_params(const shape3d &in, size_t w_width, size_t w_height,
            size_t outc, padding ptype, bool has_bias, size_t w_stride,
            size_t h_stride, size_t w_dilation, size_t h_dilation,
            const connection_table &tbl = connection_table());

    size_t in_length(size_t in_length, size_t window_size,
            padding pad_type) const;

    static size_t conv_out_dim(size_t in_width, size_t in_height,
            size_t window_size, size_t w_stride, size_t h_stride,
            size_t w_dilation, size_t h_dilation, padding pad_type);

    size_t conv_out_dim(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t w_stride, size_t h_stride,
            size_t w_dilation, size_t h_dilation, padding pad_type) const;

    void init_backend(const backend_t backend_type);

private:
    /* The convolution parameters */
    conv_params params_;

    /* Padding operation */
    Conv2dPadding padding_op_;

    /* forward op context */
    OpKernelContext fwd_ctx_;

    /* backward op context */
    OpKernelContext bwd_ctx_;

    /* Forward and backward ops */
    std::shared_ptr<OpKernel> kernel_fwd_;
    std::shared_ptr<OpKernel> kernel_back_;

    std::vector<tensor_t*> fwd_in_data_;
    std::vector<tensor_t*> bwd_in_data_;
    std::vector<tensor_t*> bwd_in_grad_;

    /* Buffer to store padded data */
    struct conv_layer_worker_specific_storage {
        tensor_t prev_out_padded_;
        tensor_t prev_delta_padded_;
    } cws_;
}
;

}  // namespace mnn
