/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/layer/layer.h"

namespace mnn {

class PartialConnectedLayer: public Layer {
public:
    typedef std::vector<std::pair<size_t, size_t>> io_connections;
    typedef std::vector<std::pair<size_t, size_t>> wi_connections;
    typedef std::vector<std::pair<size_t, size_t>> wo_connections;

    PartialConnectedLayer(size_t in_dim, size_t out_dim, size_t weight_dim,
            size_t bias_dim, Float scale_factor = Float { 1 });

    size_t param_size() const;
    size_t fan_in_size() const override;
    size_t fan_out_size() const override;

    void connect_weight(size_t input_index, size_t output_index,
            size_t weight_index);

    void connect_bias(size_t bias_index, size_t output_index);

    void forward_propagation(
            const std::vector<Matrix*> &in_data,
            std::vector<Matrix*> &out_data) override;

    void back_propagation(
            const std::vector<Matrix*> &in_data,
            const std::vector<Matrix*> &out_data,
            std::vector<Matrix*> &out_grad,
            std::vector<Matrix*> &in_grad)override;

protected:
    std::vector<io_connections> weight2io_;  // weight_id -> [(in_id, out_id)]
    std::vector<wi_connections> out2wi_;     // out_id -> [(weight_id, in_id)]
    std::vector<wo_connections> in2wo_;      // in_id -> [(weight_id, out_id)]
    std::vector<std::vector<size_t>> bias2out_;
    std::vector<size_t> out2bias_;
    Float scale_factor_;
};

}  // namespace mnn
