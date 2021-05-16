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

struct conv_layer_worker_specific_storage {
  std::vector<const vec_t *> prev_out_padded_;
  std::vector<vec_t> prev_out_buf_;
  std::vector<vec_t> prev_delta_padded_;
};

struct connection_table {
  connection_table();
  connection_table(const bool *ar, size_t rows, size_t cols);
  connection_table(size_t ngroups, size_t rows, size_t cols);

  bool is_connected(size_t x, size_t y) const;
  bool is_empty() const;

  std::deque<bool> connected_;
  size_t rows_;
  size_t cols_;
};

class conv_params : public Params {
 public:
  connection_table tbl;
  index3d<size_t> in;
  index3d<size_t> in_padded;
  index3d<size_t> out;
  index3d<size_t> weight;
  bool has_bias;
  padding pad_type;
  size_t w_stride;
  size_t h_stride;
  size_t w_dilation;
  size_t h_dilation;
};

class Conv2dPadding {
 public:
  Conv2dPadding();
  explicit Conv2dPadding(const conv_params &params);

  void copy_and_pad_input(const tensor_t &in, tensor_t &out);
  void copy_and_unpad_delta(const tensor_t &delta, tensor_t &delta_unpadded);

 private:
  conv_params params_;
};

}  // namespace mnn
