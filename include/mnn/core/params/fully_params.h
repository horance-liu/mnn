/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <stddef.h>
#include "mnn/core/params/params.h"

namespace mnn {

class fully_params : public Params {
 public:
  size_t in_size_;
  size_t out_size_;
  bool has_bias_;
};

inline fully_params &Params::fully() {
  return *(static_cast<fully_params *>(this));
}

}  // namespace mnn
