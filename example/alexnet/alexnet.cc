/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include <string>
#include "mnn/mnn.h"

namespace mnn {

class AlexNet: public Network<Sequential> {
public:
    explicit AlexNet(const std::string &name = "")
        : Network<mnn::Sequential>(name)
    {
        add(ConvolutionalLayer(224, 224, 11, 11, 3, 64, mnn::Padding::VALID, true, 4, 4));
        add(ReluLayer(54, 54, 64));
        add(AveragePoolingLayer(54, 54, 64, 2));

        add(ConvolutionalLayer(27, 27, 5, 5, 64, 192, mnn::Padding::VALID, true, 1, 1));
        add(ReluLayer(23, 23, 192));
        add(AveragePoolingLayer(23, 23, 192, 1));
        add(ConvolutionalLayer(23, 23, 3, 3, 192, 384, mnn::Padding::VALID, true, 1, 1));
        add(ReluLayer(21, 21, 384));
        add(ConvolutionalLayer(21, 21, 3, 3, 384, 256, mnn::Padding::VALID, true, 1, 1));
        add(ReluLayer(19, 19, 256));
        add(ConvolutionalLayer(19, 19, 3, 3, 256, 256, mnn::Padding::VALID, true, 1, 1));
        add(ReluLayer(17, 17, 256));
        add(AveragePoolingLayer(17, 17, 256, 1));
    }
};

} // namespace mnn

int main(int argc, char **argv)
{
    mnn::AlexNet net("alexnet");
}
