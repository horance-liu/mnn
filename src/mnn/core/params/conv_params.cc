/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnn/core/params/conv_params.h"

namespace mnn {

ConnectionTable::ConnectionTable() : rows_(0), cols_(0)
{
}
ConnectionTable::ConnectionTable(const bool *ar, size_t rows, size_t cols) : connected_(
        rows * cols), rows_(rows), cols_(cols)
{
    std::copy(ar, ar + rows * cols, connected_.begin());
}
ConnectionTable::ConnectionTable(size_t ngroups, size_t rows, size_t cols) : connected_(
        rows * cols, false), rows_(rows), cols_(cols)
{
    if (rows % ngroups || cols % ngroups) {
        throw MnnError("invalid group size");
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

bool ConnectionTable::is_connected(size_t x, size_t y) const
{
    return is_empty() ? true : connected_[y * cols_ + x];
}

bool ConnectionTable::is_empty() const
{
    return rows_ == 0 && cols_ == 0;
}

ConvParams& Params::conv()
{
    return *(static_cast<ConvParams*>(this));
}

Conv2dPadding::Conv2dPadding()
{
}

Conv2dPadding::Conv2dPadding(const ConvParams &params) : params_(params)
{
}

void Conv2dPadding::copy_and_pad_input(const Matrix &in, Matrix &out)
{
    if (params_.pad_type == Padding::VALID) {
        return;
    }

    Matrix buf(in.size());

    for_i(true, buf.size(), [&](size_t sample) {
        // alloc temporary buffer.
        buf[sample].resize(params_.in_padded.size());

        // make padded version in order to avoid corner-case in fprop/bprop
        for (size_t c = 0; c < params_.in.depth_; c++) {
            Float *pimg = &buf[sample][params_.in_padded.get_index(
                    params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
            const Float *pin = &in[sample][params_.in.get_index(0, 0, c)];

            for (size_t y = 0; y < params_.in.height_; y++) {
                std::copy(pin, pin + params_.in.width_, pimg);
                pin += params_.in.width_;
                pimg += params_.in_padded.width_;
            }
        }
    });

    out = buf;
}

void Conv2dPadding::copy_and_unpad_delta(const Matrix &delta,
        Matrix &delta_unpadded)
{
    if (params_.pad_type == Padding::VALID) {
        return;
    }

    Matrix buf(delta.size());

    for_i(true, buf.size(), [&](size_t sample) {
        // alloc temporary buffer.
        buf[sample].resize(params_.in.size());

        for (size_t c = 0; c < params_.in.depth_; c++) {
            const Float *pin = &delta[sample][params_.in_padded.get_index(
                    params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
            Float *pdst = &buf[sample][params_.in.get_index(0, 0, c)];

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
