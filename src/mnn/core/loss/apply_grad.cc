/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include "mnn/core/loss/apply_grad.h"

namespace mnn {

void apply_cost_if_defined(std::vector<Vector> &sample_gradient,
        const std::vector<Vector> &sample_cost)
{
    if (sample_gradient.size() == sample_cost.size()) {
        const size_t channel_count = sample_gradient.size();
        for (size_t channel = 0; channel < channel_count; ++channel) {
            if (sample_gradient[channel].size()
                    == sample_cost[channel].size()) {
                const size_t element_count = sample_gradient[channel].size();

                for (size_t element = 0; element < element_count; ++element) {
                    sample_gradient[channel][element] *=
                            sample_cost[channel][element];
                }
            }
        }
    }
}

}  // namespace mnn
