/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <algorithm>
#include <deque>
#include <vector>

#include "mnn/core/params/params.h"
#include "mnn/infra/util.h"

namespace mnn {

struct ConvLayerWorkerSpecificStorage {
  std::vector<const Vector *> prev_out_padded_;
  std::vector<Vector> prev_out_buf_;
  std::vector<Vector> prev_delta_padded_;
};

struct ConnectionTable {
  ConnectionTable();
  ConnectionTable(const bool *ar, size_t rows, size_t cols);
  ConnectionTable(size_t ngroups, size_t rows, size_t cols);

  bool is_connected(size_t x, size_t y) const;
  bool is_empty() const;

  std::deque<bool> connected_;
  size_t rows_;
  size_t cols_;
};

class ConvParams : public Params {
 public:
  ConnectionTable tbl;
  Index3d<size_t> in;
  Index3d<size_t> in_padded;
  Index3d<size_t> out;
  Index3d<size_t> weight;
  bool has_bias;
  Padding pad_type;
  size_t w_stride;
  size_t h_stride;
  size_t w_dilation;
  size_t h_dilation;
};

class Conv2dPadding {
 public:
  Conv2dPadding();
  explicit Conv2dPadding(const ConvParams &params);

  void copy_and_pad_input(const Matrix &in, Matrix &out);
  void copy_and_unpad_delta(const Matrix &delta, Matrix &delta_unpadded);

 private:
  ConvParams params_;
};

}  // namespace mnn
