/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include <string>
#include "mnn/mnn.h"

class alexnet: public mnn::Network<mnn::Sequential> {
public:
    explicit alexnet(const std::string &name = "")
        : mnn::Network<mnn::Sequential>(name)
    {

        using relu = mnn::activation::relu;
        using conv = mnn::layer::conv;
        using ave_pool = mnn::layer::ave_pool;

        *this   << conv(224, 224, 11, 11, 3, 64, mnn::Padding::VALID, true, 4, 4)
                << relu(54, 54, 64) << ave_pool(54, 54, 64, 2)
                << conv(27, 27, 5, 5, 64, 192, mnn::Padding::VALID, true, 1, 1)
                << relu(23, 23, 192) << ave_pool(23, 23, 192, 1)
                << conv(23, 23, 3, 3, 192, 384, mnn::Padding::VALID, true, 1, 1)
                << relu(21, 21, 384)
                << conv(21, 21, 3, 3, 384, 256, mnn::Padding::VALID, true, 1, 1)
                << relu(19, 19, 256)
                << conv(19, 19, 3, 3, 256, 256, mnn::Padding::VALID, true, 1, 1)
                << relu(17, 17, 256) << ave_pool(17, 17, 256, 1);
    }
};

int main(int argc, char **argv)
{
    alexnet net("alexnet");
}
