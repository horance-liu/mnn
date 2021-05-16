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

void parse_mnist_labels(const std::string &label_file, std::vector<Label> *labels);
void parse_mnist_images(const std::string &image_file, std::vector<Vector> *images, Float scale_min, Float scale_max, int x_padding, int y_padding);

}  // namespace mnn
