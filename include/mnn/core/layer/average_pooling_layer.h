/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/layer/partial_connected_layer.h"

namespace mnn {

class AveragePoolingLayer: public PartialConnectedLayer {
public:
    using Base = PartialConnectedLayer;

    AveragePoolingLayer(size_t in_width, size_t in_height, size_t in_channels,
            size_t pool_size, bool ceil_mode = false);

    AveragePoolingLayer(const Shape3d &in_shape, size_t pool_size,
            size_t stride, bool ceil_mode = false);

    AveragePoolingLayer(size_t in_width, size_t in_height, size_t in_channels,
            size_t pool_size, size_t stride, bool ceil_mode = false);

    AveragePoolingLayer(size_t in_width, size_t in_height, size_t in_channels,
            size_t pool_size_x, size_t pool_size_y, size_t stride_x,
            size_t stride_y, bool ceil_mode = false, Padding pad_type =
                    Padding::VALID);

    std::vector<Index3d<size_t>> in_shape() const override;
    std::vector<Index3d<size_t>> out_shape() const override;
    std::string layer_type() const override;

    void forward_propagation(
            const std::vector<Matrix*> &in_data,
            std::vector<Matrix*> &out_data) override;

    void back_propagation(
            const std::vector<Matrix*> &in_data,
            const std::vector<Matrix*> &out_data,
            std::vector<Matrix*> &out_grad,
            std::vector<Matrix*> &in_grad) override;

    std::pair<size_t, size_t> pool_size() const;

private:
    size_t stride_x_;
    size_t stride_y_;
    size_t pool_size_x_;
    size_t pool_size_y_;
    Shape3d in_;
    Shape3d out_;
    Shape3d w_;

    static size_t pool_out_dim(size_t in_size, size_t pooling_size,size_t stride);

    void init_connection(size_t pooling_size_x, size_t pooling_size_y);
    void connect_kernel(size_t pooling_size_x, size_t pooling_size_y, size_t x,size_t y, size_t inc);
};

}  // namespace mnn
