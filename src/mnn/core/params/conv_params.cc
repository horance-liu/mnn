/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/params/conv_params.h"

namespace mnn {

connection_table::connection_table() : rows_(0), cols_(0)
{
}
connection_table::connection_table(const bool *ar, size_t rows, size_t cols) : connected_(
        rows * cols), rows_(rows), cols_(cols)
{
    std::copy(ar, ar + rows * cols, connected_.begin());
}
connection_table::connection_table(size_t ngroups, size_t rows, size_t cols) : connected_(
        rows * cols, false), rows_(rows), cols_(cols)
{
    if (rows % ngroups || cols % ngroups) {
        throw nn_error("invalid group size");
    }

    size_t row_group = rows / ngroups;
    size_t col_group = cols / ngroups;

    size_t idx = 0;

    for (size_t g = 0; g < ngroups; g++) {
        for (size_t r = 0; r < row_group; r++) {
            for (size_t c = 0; c < col_group; c++) {
                idx = (r + g * row_group) * cols_ + c + g * col_group;
                connected_[idx] = true;
            }
        }
    }
}

bool connection_table::is_connected(size_t x, size_t y) const
{
    return is_empty() ? true : connected_[y * cols_ + x];
}

bool connection_table::is_empty() const
{
    return rows_ == 0 && cols_ == 0;
}

conv_params& Params::conv()
{
    return *(static_cast<conv_params*>(this));
}

Conv2dPadding::Conv2dPadding()
{
}

Conv2dPadding::Conv2dPadding(const conv_params &params) : params_(params)
{
}

void Conv2dPadding::copy_and_pad_input(const tensor_t &in, tensor_t &out)
{
    if (params_.pad_type == padding::valid) {
        return;
    }

    tensor_t buf(in.size());

    for_i(true, buf.size(), [&](size_t sample) {
        // alloc temporary buffer.
        buf[sample].resize(params_.in_padded.size());

        // make padded version in order to avoid corner-case in fprop/bprop
        for (size_t c = 0; c < params_.in.depth_; c++) {
            float_t *pimg = &buf[sample][params_.in_padded.get_index(
                    params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
            const float_t *pin = &in[sample][params_.in.get_index(0, 0, c)];

            for (size_t y = 0; y < params_.in.height_; y++) {
                std::copy(pin, pin + params_.in.width_, pimg);
                pin += params_.in.width_;
                pimg += params_.in_padded.width_;
            }
        }
    });

    out = buf;
}

void Conv2dPadding::copy_and_unpad_delta(const tensor_t &delta,
        tensor_t &delta_unpadded)
{
    if (params_.pad_type == padding::valid) {
        return;
    }

    tensor_t buf(delta.size());

    for_i(true, buf.size(), [&](size_t sample) {
        // alloc temporary buffer.
        buf[sample].resize(params_.in.size());

        for (size_t c = 0; c < params_.in.depth_; c++) {
            const float_t *pin = &delta[sample][params_.in_padded.get_index(
                    params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
            float_t *pdst = &buf[sample][params_.in.get_index(0, 0, c)];

            for (size_t y = 0; y < params_.in.height_; y++) {
                std::copy(pin, pin + params_.in.width_, pdst);
                pdst += params_.in.width_;
                pin += params_.in_padded.width_;
            }
        }
    });

    delta_unpadded = buf;
}

}  // namespace mnn
