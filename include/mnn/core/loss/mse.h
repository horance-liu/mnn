/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/util.h"

namespace mnn {

class mse {
 public:
  static float_t f(const vec_t &y, const vec_t &t);
  static vec_t df(const vec_t &y, const vec_t &t);
};

}  // namespace mnn
