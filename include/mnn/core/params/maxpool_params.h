/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <vector>

#include "mnn/core/params/params.h"

namespace mnn {

class maxpool_params : public Params {
 public:
  index3d<size_t> in;
  index3d<size_t> out;
  size_t pool_size_x;
  size_t pool_size_y;
  size_t stride_x;
  size_t stride_y;
  bool ceil_mode;
  padding pad_type;

  std::vector<std::vector<size_t>> out2inmax;
  std::vector<std::vector<size_t>> out2in;
  std::vector<size_t> in2out;
};

// TODO(nyanp): can we do better here?
inline maxpool_params &Params::maxpool() {
  return *(static_cast<maxpool_params *>(this));
}

}  // namespace mnn
