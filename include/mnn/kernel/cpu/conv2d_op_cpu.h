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

void conv2d_op_internal(const Matrix &in_data, const Vector &W,
        const Vector &bias, Matrix &out_data, const ConvParams &params,
        const bool parallelize);

/******************************************************************/

template<typename Matrix, typename Vector>
void conv2d_op_internal(const Matrix &prev_out, const Vector &W, Matrix &dW,
        Matrix &db, Matrix &curr_delta, Matrix &prev_delta,
        const ConvParams &params, const bool parallelize)
{
    typedef typename Vector::value_type Float;

    for_i(parallelize, prev_out.size(), [&](size_t sample) {
    // propagate delta to previous layer
        for (size_t inc = 0; inc < params.in.depth_; inc++) {
            for (size_t outc = 0; outc < params.out.depth_; outc++) {
                if (!params.tbl.is_connected(outc, inc)) continue;

                size_t idx = 0;
                idx = params.in.depth_ * outc + inc;
                idx = params.weight.get_index(0, 0, idx);
                const Float *pw = &W[idx];

                idx = params.out.get_index(0, 0, outc);
                const Float *pdelta_src = &curr_delta[sample][idx];

                idx = params.in_padded.get_index(0, 0, inc);
                // Float* pdelta_dst = &(*prev_delta)[sample][idx];
                Float *pdelta_dst = &prev_delta[sample][idx];

                for (size_t y = 0; y < params.out.height_; y++) {
                    for (size_t x = 0; x < params.out.width_; x++) {
                        const Float *ppw = pw;

                        idx = y * params.out.width_ + x;
                        const Float ppdelta_src = pdelta_src[idx];

                        Float *ppdelta_dst =
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
                        Float dst {0};

                        size_t idx = 0;
                        idx = params.in_padded.get_index(wx, wy, inc);
                        const Float *prevo = &prev_out[sample][idx];

                        idx = params.out.get_index(0, 0, outc);
                        const Float *delta = &curr_delta[sample][idx];

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
                const Float *delta = &curr_delta[sample][idx];
                const Float *deltaa = delta + params.out.width_ * params.out.height_;
                db[sample][outc] += std::accumulate(delta, deltaa, Float {0});
            }
        }
    });
}

}  // namespace kernels
}  // namespace mnn
