/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/util.h"
#include "mnn/core/params/conv_params.h"

namespace mnn {
namespace kernels {

void conv2d_op_internal(const tensor_t &in_data, const vec_t &W,
        const vec_t &bias, tensor_t &out_data, const conv_params &params,
        const bool parallelize);

/******************************************************************/

template<typename tensor_t, typename vec_t>
void conv2d_op_internal(const tensor_t &prev_out, const vec_t &W, tensor_t &dW,
        tensor_t &db, tensor_t &curr_delta, tensor_t &prev_delta,
        const conv_params &params, const bool parallelize)
{
    typedef typename vec_t::value_type float_t;

    for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // propagate delta to previous layer
        for (size_t inc = 0; inc < params.in.depth_; inc++) {
            for (size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                size_t idx = 0;
                idx = params.in.depth_ * outc + inc;
                idx = params.weight.get_index(0, 0, idx);
                const float_t *pw = &W[idx];

                idx = params.out.get_index(0, 0, outc);
                const float_t *pdelta_src = &curr_delta[sample][idx];

                idx = params.in_padded.get_index(0, 0, inc);
                // float_t* pdelta_dst = &(*prev_delta)[sample][idx];
                float_t *pdelta_dst = &prev_delta[sample][idx];

                for (size_t y = 0; y < params.out.height_; y++) {
                    for (size_t x = 0; x < params.out.width_; x++) {
                        const float_t *ppw = pw;

                        idx = y * params.out.width_ + x;
                        const float_t ppdelta_src = pdelta_src[idx];

                        float_t *ppdelta_dst =
                        pdelta_dst + y * params.h_stride * params.in_padded.width_ +
                        x * params.w_stride;

                        for (size_t wy = 0; wy < params.weight.height_; wy++) { // NOLINT
                            for (size_t wx = 0; wx < params.weight.width_; wx++) { // NOLINT
                                idx = wy * params.in_padded.width_ + wx;
                                ppdelta_dst[idx] += *ppw++ * ppdelta_src;
                            }
                        }
                    }
                }
            }
        }

        // accumulate dw
        for (size_t inc = 0; inc < params.in.depth_; inc++) {
            for (size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                for (size_t wy = 0; wy < params.weight.height_; wy++) {
                    for (size_t wx = 0; wx < params.weight.width_; wx++) {
                        float_t dst {0};

                        size_t idx = 0;
                        idx = params.in_padded.get_index(wx, wy, inc);
                        const float_t *prevo = &prev_out[sample][idx];

                        idx = params.out.get_index(0, 0, outc);
                        const float_t *delta = &curr_delta[sample][idx];

                        if (params.w_stride > 1) {
                            for (size_t y = 0; y < params.out.height_; y++) {
                                size_t prevo_idx =
                                y * params.in_padded.width_ * params.h_stride;
                                size_t delta_idx = y * params.out.width_;

                                for (size_t x = 0; x < params.out.width_; x++) {
                                    dst += prevo[prevo_idx + x * params.w_stride] *
                                    delta[delta_idx + x];
                                }
                            }
                        } else {
                            for (size_t y = 0; y < params.out.height_; y++) {
                                dst += vectorize::dot(
                                        prevo + y * params.in_padded.width_ * params.h_stride,
                                        delta + y * params.out.width_, params.out.width_);
                            }
                        }

                        idx = params.in.depth_ * outc + inc;
                        dW[sample][params.weight.get_index(wx, wy, idx)] += dst;
                    }
                }
            }
        }

        // accumulate db
        if (params.has_bias) {
            for (size_t outc = 0; outc < params.out.depth_; outc++) {
                size_t idx = params.out.get_index(0, 0, outc);
                const float_t *delta = &curr_delta[sample][idx];
                const float_t *deltaa = delta + params.out.width_ * params.out.height_;
                db[sample][outc] += std::accumulate(delta, deltaa, float_t {0});
            }
        }
    });
}

}  // namespace kernels
}  // namespace mnn
