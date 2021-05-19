/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/kernel/cpu/conv2d_op_cpu.h"

namespace mnn {
namespace kernels {

void conv2d_op_internal(const Matrix &in_data, const Vector &W,
        const Vector &bias, Matrix &out_data, const ConvParams &params,
        const bool parallelize)
{
    for_(parallelize, 0u, in_data.size(), [&](const BlockedRange &r) {
        size_t out_area = params.out.area();
        size_t iw = params.in_padded.width_;
        size_t id = params.in.depth_;
        size_t ow = params.out.width_;
        size_t oh = params.out.height_;
        size_t od = params.out.depth_;
        size_t kw = params.weight.width_;
        size_t kh = params.weight.height_;
        size_t w_dilation = params.w_dilation;
        size_t h_dilation = params.h_dilation;
        size_t elem_stride = params.w_stride;
        size_t line_stride = iw * params.h_stride;
        for (size_t sample = r.begin(); sample < r.end(); sample++) {
            const Vector &in = in_data[sample];
            Vector &a = out_data[sample];
            for (size_t o = 0; o < od; o++) {
                Float *pa = &a[params.out.get_index(0, 0, o)];
                for (size_t inc = 0; inc < id; inc++) {
                    if (!params.tbl.is_connected(o, inc)) continue;
                    size_t idx;
                    idx = params.weight.get_index(0, 0, id * o + inc);
                    const Float *pw = &W[idx];
                    idx = params.in_padded.get_index(0, 0, inc);
                    const Float *pin = &in[idx];
                    Float *pout = pa;
                    for (size_t y = 0; y < oh; y++) {
                        const Float *pin_line = pin;
                        for (size_t x = 0; x < ow; x++) {
                            const Float *pin_element = pin_line;
                            const Float *pw_element = pw;
                            Float sum {0};
                            // should be optimized for small kernel(3x3,5x5)
                            for (size_t wy = 0; wy < kh; wy++) {    // NOLINT
                                for (size_t wx = 0; wx < kw; wx++) {  // NOLINT
                                    sum += pw_element[wx] * pin_element[wx * w_dilation];
                                }
                                pw_element += kw;
                                pin_element += iw * h_dilation;
                            }
                            pout[x] += sum;
                            pin_line += elem_stride;
                        }
                        pout += ow;
                        pin += line_stride;
                    }
                }
                if (params.has_bias) {
                    vectorize::add(bias[o], out_area, pa);
                }
            }
        }
    }, 0u);
}

}  // namespace kernels
}  // namespace mnn
