/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/node.h"
#include "mnn/core/layer/layer.h"
#include <vector>

#include "mnn/core/params/conv_params.h"
#include "mnn/core/params/fully_params.h"
#include "mnn/core/params/maxpool_params.h"

namespace mnn {

class context;

enum class BackendType { CPU, GPU, ASCI};

inline std::ostream &operator<<(std::ostream &os, BackendType type) {
  switch (type) {
    case BackendType::CPU: os << "CPU"; break;
    case BackendType::GPU: os << "GPU"; break;
    case BackendType::ASCI: os << "ASCI"; break;
    default: throw MnnError("Not supported device."); break;
  }
  return os;
}

inline BackendType default_engine() {
  return BackendType::CPU;
}

}  // namespace mnn
