/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/layer/layer.h"
#include "mnn/core/graph/op_kernel.h"
#include "mnn/infra/backend.h"

namespace mnn {

class ConvolutionalLayer: public Layer {
public:
    ConvolutionalLayer(size_t in_width, size_t in_height, size_t window_size,
            size_t in_channels, size_t out_channels, Padding pad_type =
                    Padding::VALID, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            BackendType backend_type = default_engine());

    ConvolutionalLayer(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t in_channels, size_t out_channels,
            Padding pad_type = Padding::VALID, bool has_bias = true,
            size_t w_stride = 1, size_t h_stride = 1, size_t w_dilation = 1,
            size_t h_dilation = 1, BackendType backend_type = default_engine());

    ConvolutionalLayer(size_t in_width, size_t in_height, size_t window_size,
            size_t in_channels, size_t out_channels,
            const ConnectionTable &connection_table, Padding pad_type =
                    Padding::VALID, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            BackendType backend_type = default_engine());

    ConvolutionalLayer(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t in_channels, size_t out_channels,
            const ConnectionTable &connection_table, Padding pad_type =
                    Padding::VALID, bool has_bias = true, size_t w_stride = 1,
            size_t h_stride = 1, size_t w_dilation = 1, size_t h_dilation = 1,
            BackendType backend_type = default_engine());

    ConvolutionalLayer(ConvolutionalLayer &&other);

    size_t fan_in_size() const override;
    size_t fan_out_size() const override;

    void forward_propagation(
            const std::vector<Matrix*> &in_data,
            std::vector<Matrix*> &out_data) override;

    void back_propagation(
            const std::vector<Matrix*> &in_data,
            const std::vector<Matrix*> &out_data,
            std::vector<Matrix*> &out_grad,
            std::vector<Matrix*> &in_grad) override;

    void set_sample_count(size_t sample_count) override;
    std::string layer_type() const override;

    std::vector<Index3d<size_t>> in_shape() const override;
    std::vector<Index3d<size_t>> out_shape() const override;

private:
    Matrix* in_data_padded(const std::vector<Matrix*> &in);
    void conv_set_params(const Shape3d &in, size_t w_width, size_t w_height,
            size_t outc, Padding ptype, bool has_bias, size_t w_stride,
            size_t h_stride, size_t w_dilation, size_t h_dilation,
            const ConnectionTable &tbl = ConnectionTable());

    size_t in_length(size_t in_length, size_t window_size,
            Padding pad_type) const;

    static size_t conv_out_dim(size_t in_width, size_t in_height,
            size_t window_size, size_t w_stride, size_t h_stride,
            size_t w_dilation, size_t h_dilation, Padding pad_type);

    size_t conv_out_dim(size_t in_width, size_t in_height, size_t window_width,
            size_t window_height, size_t w_stride, size_t h_stride,
            size_t w_dilation, size_t h_dilation, Padding pad_type) const;

    void init_backend(const BackendType backend_type);

private:
    /* The convolution parameters */
    ConvParams params_;

    /* Padding operation */
    Conv2dPadding padding_op_;

    /* forward op context */
    OpKernelContext fwd_ctx_;

    /* backward op context */
    OpKernelContext bwd_ctx_;

    /* Forward and backward ops */
    std::shared_ptr<OpKernel> kernel_fwd_;
    std::shared_ptr<OpKernel> kernel_back_;

    std::vector<Matrix*> fwd_in_data_;
    std::vector<Matrix*> bwd_in_data_;
    std::vector<Matrix*> bwd_in_grad_;

    /* Buffer to store padded data */
    struct conv_layer_worker_specific_storage {
        Matrix prev_out_padded_;
        Matrix prev_delta_padded_;
    } cws_;
}
;

}  // namespace mnn
