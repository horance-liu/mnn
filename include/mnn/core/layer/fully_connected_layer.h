/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/op_kernel.h"
#include "mnn/core/layer/layer.h"

namespace mnn {

class fully_connected_layer: public layer {
public:
    fully_connected_layer(
            size_t in_dim, size_t out_dim, bool has_bias = true,
            backend_t backend_type = default_engine());

    fully_connected_layer(fully_connected_layer &&other);

    size_t fan_in_size() const override;
    size_t fan_out_size() const override;

    std::vector<index3d<size_t>> in_shape() const override;
    std::vector<index3d<size_t>> out_shape() const override;

    void forward_propagation(
            const std::vector<tensor_t*> &in_data,
            std::vector<tensor_t*> &out_data) override;

    void back_propagation(
            const std::vector<tensor_t*> &in_data,
            const std::vector<tensor_t*> &out_data,
            std::vector<tensor_t*> &out_grad,
            std::vector<tensor_t*> &in_grad) override;

    std::string layer_type() const override;

protected:
    void set_params(const size_t in_size, const size_t out_size, bool has_bias);
    void init_backend(backend_t backend_type);

private:
    fully_params params_;
    OpKernelContext fwd_ctx_;
    OpKernelContext bwd_ctx_;

    std::shared_ptr<OpKernel> kernel_fwd_;
    std::shared_ptr<OpKernel> kernel_back_;
};

}  // namespace mnn
