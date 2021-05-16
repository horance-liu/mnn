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

class FullyParams : public Params {
 public:
  size_t in_size_;
  size_t out_size_;
  bool has_bias_;
};

inline FullyParams &Params::fully() {
  return *(static_cast<FullyParams *>(this));
}

}  // namespace mnn
