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

class Mse {
 public:
  static Float f(const Vector &y, const Vector &t);
  static Vector df(const Vector &y, const Vector &t);
};

}  // namespace mnn
