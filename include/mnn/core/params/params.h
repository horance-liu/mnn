/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#pragma once

namespace mnn {

class ConvParams;
class FullyParams;
class MaxpoolParams;

class Params {
 public:
  Params() {}

  ConvParams &conv();
  FullyParams &fully();
  MaxpoolParams &maxpool();
};

}  // namespace mnn
