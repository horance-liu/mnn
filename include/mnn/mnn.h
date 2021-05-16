/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/infra/config.h"
#include "mnn/core/graph/network.h"
#include "mnn/core/graph/nodes.h"
#include "mnn/core/graph/tensor.h"

#include "mnn/core/activation/relu_layer.h"
#include "mnn/core/activation/sigmoid_layer.h"
#include "mnn/core/activation/softmax_layer.h"
#include "mnn/core/activation/tanh_layer.h"

#include "mnn/core/layer/average_pooling_layer.h"
#include "mnn/core/layer/convolutional_layer.h"
#include "mnn/core/layer/fully_connected_layer.h"

#include "mnn/core/loss/mse.h"
#include "mnn/core/loss/cross_entropy.h"

#include "mnn/core/optimizer/adam.h"
#include "mnn/core/optimizer/adagrad.h"
#include "mnn/core/optimizer/gradient_descent.h"

#include "mnn/infra/product.h"
#include "mnn/infra/weight_init.h"
#include "mnn/infra/text_progress.h"
#include "mnn/infra/timer.h"

namespace mnn {
namespace layers {

using conv = mnn::convolutional_layer;
using ave_pool = mnn::average_pooling_layer;
using fc = mnn::fully_connected_layer;

}  // namespace layers

namespace activation {

using sigmoid = mnn::sigmoid_layer;
using tanh = mnn::tanh_layer;
using relu = mnn::relu_layer;
using softmax = mnn::softmax_layer;

}  // namespace activation

}  // namespace mnn

