/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/core/layer/average_pooling_layer.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "mnn/infra/util.h"

namespace mnn {

void average_pooling_kernel(bool parallelize,
        const std::vector<Matrix*> &in_data, std::vector<Matrix*> &out_data,
        const Shape3d &out_dim, Float scale_factor,
        std::vector<typename PartialConnectedLayer::wi_connections> &out2wi)
{
    for_i(parallelize, in_data[0]->size(),
            [&](
                    size_t sample) {
                        const Vector &in = (*in_data[0])[sample];
                        const Vector &W = (*in_data[1])[0];
                        const Vector &b = (*in_data[2])[0];
                        Vector &out = (*out_data[0])[sample];

                        auto oarea = out_dim.area();
                        size_t idx = 0;
                        for (size_t d = 0; d < out_dim.depth_; ++d) {
                            Float weight = W[d] * scale_factor;
                            Float bias = b[d];
                            for (size_t i = 0; i < oarea; ++i, ++idx) {
                                const auto &connections = out2wi[idx];
                                Float value {0};
                                for (auto connection : connections) value += in[connection.second];
                                value *= weight;
                                value += bias;
                                out[idx] = value;
                            }
                        }

                        assert(out.size() == out2wi.size());
                    });
}

void average_pooling_back_kernel(bool parallelize,
        const std::vector<Matrix*> &in_data,
        const std::vector<Matrix*> &out_data,
        std::vector<Matrix*> &out_grad, std::vector<Matrix*> &in_grad,
        const Shape3d &in_dim, Float scale_factor,
        std::vector<typename PartialConnectedLayer::io_connections> &weight2io,
        std::vector<typename PartialConnectedLayer::wo_connections> &in2wo,
        std::vector<std::vector<size_t>> &bias2out)
{
    MNN_UNREFERENCED_PARAMETER(out_data);
    for_i(parallelize, in_data[0]->size(), [&](size_t sample) {
        const Vector &prev_out = (*in_data[0])[sample];
        const Vector &W = (*in_data[1])[0];
        Vector &dW = (*in_grad[1])[sample];
        Vector &db = (*in_grad[2])[sample];
        Vector &prev_delta = (*in_grad[0])[sample];
        Vector &curr_delta = (*out_grad[0])[sample];

        auto inarea = in_dim.area();
        size_t idx = 0;
        for (size_t i = 0; i < in_dim.depth_; ++i) {
            Float weight = W[i] * scale_factor;
            for (size_t j = 0; j < inarea; ++j, ++idx) {
                prev_delta[idx] = weight * curr_delta[in2wo[idx][0].second];
            }
        }

        for (size_t i = 0; i < weight2io.size(); ++i) {
            const auto &connections = weight2io[i];
            Float diff {0};

            for (auto connection : connections)
            diff += prev_out[connection.first] * curr_delta[connection.second];

            dW[i] += diff * scale_factor;
        }

        for (size_t i = 0; i < bias2out.size(); i++) {
            const std::vector<size_t> &outs = bias2out[i];
            Float diff {0};

            for (auto o : outs) diff += curr_delta[o];

            db[i] += diff;
        }
    });
}

AveragePoolingLayer::AveragePoolingLayer(size_t in_width, size_t in_height,
        size_t in_channels, size_t pool_size, bool ceil_mode) : AveragePoolingLayer(
        in_width, in_height, in_channels, pool_size,
        (in_height == 1 ? 1 : pool_size), ceil_mode)
{
}

AveragePoolingLayer::AveragePoolingLayer(const Shape3d &in_shape,
        size_t pool_size, size_t stride, bool ceil_mode) : AveragePoolingLayer(
        in_shape.width_, in_shape.width_, in_shape.depth_, pool_size, stride,
        ceil_mode)
{
}

AveragePoolingLayer::AveragePoolingLayer(size_t in_width, size_t in_height,
        size_t in_channels, size_t pool_size, size_t stride, bool ceil_mode) : AveragePoolingLayer(
        in_width, in_height, in_channels, pool_size,
        (in_height == 1 ? 1 : pool_size), stride, stride, ceil_mode,
        Padding::VALID)
{
}

AveragePoolingLayer::AveragePoolingLayer(size_t in_width, size_t in_height,
        size_t in_channels, size_t pool_size_x, size_t pool_size_y,
        size_t stride_x, size_t stride_y, bool ceil_mode, Padding pad_type)
    : Base(
              in_width * in_height * in_channels,
              pool_out_length(in_width, pool_size_x, stride_x, ceil_mode, pad_type) *
              pool_out_length(in_height, pool_size_y, stride_y, ceil_mode,pad_type) * in_channels,
              in_channels,
              in_channels,
              Float(1) / (pool_size_x * pool_size_y)
         ),

        stride_x_(stride_x),
        stride_y_(stride_y),
        pool_size_x_(pool_size_x),
        pool_size_y_(pool_size_y),
        in_(in_width, in_height, in_channels),

        out_(
                pool_out_length(in_width, pool_size_x, stride_x, ceil_mode, pad_type),
                pool_out_length(in_height, pool_size_y, stride_y, ceil_mode, pad_type),
                in_channels
            ),

        w_(pool_size_x, pool_size_y, in_channels)
{
    if ((in_width % pool_size_x) || (in_height % pool_size_y)) {
        pooling_size_mismatch(in_width, in_height, pool_size_x, pool_size_y);
    }

    init_connection(pool_size_x, pool_size_y);
}

std::vector<Index3d<size_t>> AveragePoolingLayer::in_shape() const

{
    return {in_, w_, Index3d<size_t>(1, 1, out_.depth_)};
}

std::vector<Index3d<size_t>> AveragePoolingLayer::out_shape() const

{
    return {out_};
}

std::string AveragePoolingLayer::layer_type() const
{
    return "ave-pool";
}

void AveragePoolingLayer::forward_propagation(
        const std::vector<Matrix*> &in_data, std::vector<Matrix*> &out_data)
{
    average_pooling_kernel(parallelize_, in_data, out_data, out_,
            Base::scale_factor_, Base::out2wi_);
}

void AveragePoolingLayer::back_propagation(
        const std::vector<Matrix*> &in_data,
        const std::vector<Matrix*> &out_data,
        std::vector<Matrix*> &out_grad, std::vector<Matrix*> &in_grad)

{
    average_pooling_back_kernel(parallelize_, in_data, out_data, out_grad,
            in_grad, in_, Base::scale_factor_, Base::weight2io_, Base::in2wo_,
            Base::bias2out_);
}

std::pair<size_t, size_t> AveragePoolingLayer::pool_size() const
{
    return std::make_pair(pool_size_x_, pool_size_y_);
}

size_t AveragePoolingLayer::pool_out_dim(size_t in_size, size_t pooling_size,
        size_t stride)
{
    return static_cast<int>(std::ceil(
            (static_cast<Float>(in_size) - pooling_size) / stride) + 1);
}

void AveragePoolingLayer::init_connection(size_t pooling_size_x,
        size_t pooling_size_y)
{
    for (size_t c = 0; c < in_.depth_; ++c) {
        for (size_t y = 0; y < in_.height_ - pooling_size_y + 1; y +=
                stride_y_) {
            for (size_t x = 0; x < in_.width_ - pooling_size_x + 1; x +=
                    stride_x_) {
                connect_kernel(pooling_size_x, pooling_size_y, x, y, c);
            }
        }
    }

    for (size_t c = 0; c < in_.depth_; ++c) {
        for (size_t y = 0; y < out_.height_; ++y) {
            for (size_t x = 0; x < out_.width_; ++x) {
                this->connect_bias(c, out_.get_index(x, y, c));
            }
        }
    }
}

void AveragePoolingLayer::connect_kernel(size_t pooling_size_x,
        size_t pooling_size_y, size_t x, size_t y, size_t inc)
{
    size_t dymax = std::min(pooling_size_y, in_.height_ - y);
    size_t dxmax = std::min(pooling_size_x, in_.width_ - x);
    size_t dstx = x / stride_x_;
    size_t dsty = y / stride_y_;
    size_t outidx = out_.get_index(dstx, dsty, inc);
    for (size_t dy = 0; dy < dymax; ++dy) {
        for (size_t dx = 0; dx < dxmax; ++dx) {
            this->connect_weight(in_.get_index(x + dx, y + dy, inc), outidx,
                    inc);
        }
    }
}

}  // namespace mnn
