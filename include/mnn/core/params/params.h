/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#pragma once

namespace mnn {

class conv_params;
class fully_params;
class maxpool_params;

class Params {
 public:
  Params() {}

  conv_params &conv();
  fully_params &fully();
  maxpool_params &maxpool();
};

}  // namespace mnn
