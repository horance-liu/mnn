/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <vector>

#include "mnn/core/params/conv_params.h"
#include "mnn/core/params/fully_params.h"
#include "mnn/core/params/maxpool_params.h"
#include "mnn/core/layer/layer.h"
#include "mnn/core/graph/node.h"

namespace mnn {

class context;

enum class backend_t { cpu, gpu };

inline std::ostream &operator<<(std::ostream &os, backend_t type) {
  switch (type) {
    case backend_t::cpu: os << "CPU"; break;
    case backend_t::gpu: os << "GPU"; break;
    default: throw nn_error("Not supported device."); break;
  }
  return os;
}

inline backend_t default_engine() {
  return backend_t::cpu;
}

}  // namespace mnn
