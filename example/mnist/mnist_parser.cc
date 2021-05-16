/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#include "mnist_parser.h"

#include "mnn/infra/util.h"
#include <fstream>

namespace mnn {

struct MnistHeader {
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
};

static inline void parse_mnist_header(std::ifstream &ifs, MnistHeader &header)
{
    ifs.read(reinterpret_cast<char*>(&header.magic_number), 4);
    ifs.read(reinterpret_cast<char*>(&header.num_items), 4);
    ifs.read(reinterpret_cast<char*>(&header.num_rows), 4);
    ifs.read(reinterpret_cast<char*>(&header.num_cols), 4);

    if (is_little_endian()) {
        reverse_endian(&header.magic_number);
        reverse_endian(&header.num_items);
        reverse_endian(&header.num_rows);
        reverse_endian(&header.num_cols);
    }

    if (header.magic_number != 0x00000803 || header.num_items <= 0)
        throw MnnError("MNIST label-file format error");
    if (ifs.fail() || ifs.bad())
        throw MnnError("file error");
}

void parse_mnist_image(std::ifstream &ifs, const MnistHeader &header,
        Float scale_min, Float scale_max, int x_padding, int y_padding,
        Vector &dst)
{
    const int width = header.num_cols + 2 * x_padding;
    const int height = header.num_rows + 2 * y_padding;

    std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);

    ifs.read(reinterpret_cast<char*>(&image_vec[0]),
            header.num_rows * header.num_cols);

    dst.resize(width * height, scale_min);

    for (uint32_t y = 0; y < header.num_rows; y++) {
        for (uint32_t x = 0; x < header.num_cols; x++) {
            dst[width * (y + y_padding) + x + x_padding] = (image_vec[y
                    * header.num_cols + x] / Float(255))
                    * (scale_max - scale_min) + scale_min;
        }
    }
}

void parse_mnist_labels(const std::string &label_file,
        std::vector<Label> *labels)
{
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw MnnError("failed to open file:" + label_file);

    uint32_t magic_number, num_items;

    ifs.read(reinterpret_cast<char*>(&magic_number), 4);
    ifs.read(reinterpret_cast<char*>(&num_items), 4);

    if (is_little_endian()) {  // MNIST data is big-endian format
        reverse_endian(&magic_number);
        reverse_endian(&num_items);
    }

    if (magic_number != 0x00000801 || num_items <= 0)
        throw MnnError("MNIST label-file format error");

    labels->resize(num_items);
    for (uint32_t i = 0; i < num_items; i++) {
        uint8_t label;
        ifs.read(reinterpret_cast<char*>(&label), 1);
        (*labels)[i] = static_cast<Label>(label);
    }
}

void parse_mnist_images(const std::string &image_file,
        std::vector<Vector> *images, Float scale_min, Float scale_max,
        int x_padding, int y_padding)
{
    if (x_padding < 0 || y_padding < 0)
        throw MnnError("padding size must not be negative");
    if (scale_min >= scale_max)
        throw MnnError("scale_max must be greater than scale_min");

    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw MnnError("failed to open file:" + image_file);

    MnistHeader header;

    parse_mnist_header(ifs, header);

    images->resize(header.num_items);
    for (uint32_t i = 0; i < header.num_items; i++) {
        Vector image;
        parse_mnist_image(ifs, header, scale_min, scale_max, x_padding,
                y_padding, image);
        (*images)[i] = image;
    }
}

}  // namespace mnn
