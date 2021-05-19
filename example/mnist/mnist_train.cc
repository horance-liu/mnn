/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */
#include <iostream>
#include <array>
#include "mnn/mnn.h"
#include "mnist_parser.h"

static void construct_net(mnn::Network<mnn::Sequential> &nn,
        mnn::BackendType backend_type)
{
// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    // clang-format off
    static const bool tbl[] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O };
// clang-format on
#undef O
#undef X

    nn.add(mnn::ConvolutionalLayer(
              32, 32, 5, 1, 6,   // C1, 1@32x32-in, 6@28x28-out
              mnn::Padding::VALID, true, 1, 1, 1, 1, backend_type
          ));

    nn.add(mnn::TanhLayer());

    nn.add(mnn::AveragePoolingLayer(28, 28, 6, 2)); // S2, 6@28x28-in, 6@14x14-out

    nn.add(mnn::TanhLayer());

    nn.add(mnn::ConvolutionalLayer(
              14, 14, 5, 6, 16,   // C3, 6@14x14-in, 16@10x10-out
              mnn::ConnectionTable(tbl, 6, 16), mnn::Padding::VALID, true, 1, 1, 1, 1, backend_type));

    nn.add(mnn::TanhLayer());

    nn.add(mnn::AveragePoolingLayer(10, 10, 16, 2)); // S4, 16@10x10-in, 16@5x5-out

    nn.add(mnn::TanhLayer());

    nn.add(mnn::ConvolutionalLayer(5, 5, 5, 16, 120,   // C5, 16@5x5-in, 120@1x1-out
            mnn::Padding::VALID, true, 1, 1, 1, 1, backend_type));

    nn.add(mnn::TanhLayer());

    nn.add(mnn::FullyConnectedLayer(120, 10, true, backend_type));  // F6, 120-in, 10-out
    nn.add(mnn::TanhLayer());
}

static void train_lenet(const std::string &data_dir_path, double learning_rate,
        const int n_train_epochs, const int n_minibatch,
        mnn::BackendType backend_type)
{
    // specify loss-function and learning strategy
    mnn::Network<mnn::Sequential> nn("lenet");
    mnn::Adagrad optimizer;

    construct_net(nn, backend_type);

    std::cout << "start loading dataset..." << std::endl;

    // load MNIST dataset
    std::vector<mnn::Label> train_labels, test_labels;
    std::vector<mnn::Vector> train_images, test_images;

    mnn::parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
            &train_labels);
    mnn::parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
            &train_images, -1.0, 1.0, 2, 2);
    mnn::parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
            &test_labels);
    mnn::parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
            &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training model..." << std::endl;

    mnn::TextProgress disp(train_images.size());
    mnn::Timer t;

    optimizer.alpha *= std::min(mnn::Float(4),
            static_cast<mnn::Float>(sqrt(n_minibatch) * learning_rate));

    int epoch = 1;
    // create callback
    auto on_enumerate_epoch = [&]() {
        std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
        << t.elapsed() << "s elapsed." << std::endl;
        ++epoch;
        mnn::Result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        if (epoch <= n_train_epochs) {
            disp.restart(train_images.size());
        }

        t.restart();
    };

    auto on_enumerate_minibatch = [&]() {disp += n_minibatch;};

    // training
    nn.train<mnn::Mse>(optimizer, train_images, train_labels, n_minibatch,
            n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training model." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);
}

static mnn::BackendType parse_backend_name(const std::string &name)
{
    const std::array<const std::string, 5> names = { { "cpu", "gpu", } };
    for (size_t i = 0; i < names.size(); ++i) {
        if (name.compare(names[i]) == 0) {
            return static_cast<mnn::BackendType>(i);
        }
    }
    return mnn::default_engine();
}

static void usage(const char *argv0)
{
    std::cout << "Usage: " << argv0
            << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epoch 2"
            << " --batch_size 16"
            << " --backend cpu"
            << std::endl;
}

int main(int argc, char **argv)
{
    double learning_rate = 1;
    int epochs = 2;
    int minibatch_size = 16;
    std::string data_path = "";

    mnn::BackendType backend_type = mnn::default_engine();

    if (argc == 2) {
        std::string argname(argv[1]);
        if (argname == "--help" || argname == "-h") {
            usage(argv[0]);
            return 0;
        }
    }

    for (int count = 1; count + 1 < argc; count += 2) {
        std::string argname(argv[count]);
        if (argname == "--learning_rate") {
            learning_rate = atof(argv[count + 1]);
        } else if (argname == "--epoch") {
            epochs = atoi(argv[count + 1]);
        } else if (argname == "--batch_size") {
            minibatch_size = atoi(argv[count + 1]);
        } else if (argname == "--backend") {
            backend_type = parse_backend_name(argv[count + 1]);
        } else if (argname == "--data_path") {
            data_path = std::string(argv[count + 1]);
        } else {
            std::cerr << "Invalid parameter specified - \"" << argname << "\""
                      << std::endl;
            usage(argv[0]);
            return -1;
        }
    }

    if (data_path == "") {
        std::cerr << "Data path not specified." << std::endl;
        usage(argv[0]);
        return -1;
    }

    if (learning_rate <= 0) {
        std::cerr
                << "Invalid learning rate. The leaarning rate must be greater than 0."
                << std::endl;
        return -1;
    }

    if (epochs <= 0) {
        std::cerr << "Invalid number of epochs. The number of epochs must be "
                "greater than 0." << std::endl;
        return -1;
    }

    if (minibatch_size <= 0 || minibatch_size > 60000) {
        std::cerr
                << "Invalid minibatch size. The minibatch size must be greater than 0"
                        " and less than dataset size (60000)." << std::endl;
        return -1;
    }

    std::cout << "Running with the following parameters:" << std::endl
            << "Data path: "        << data_path      << std::endl
            << "Learning rate: "    << learning_rate  << std::endl
            << "Minibatch size: "   << minibatch_size << std::endl
            << "Number of epochs: " << epochs         << std::endl
            << "Backend type: "     << backend_type   << std::endl;
    try {
        train_lenet(data_path, learning_rate, epochs, minibatch_size, backend_type);
    } catch (mnn::MnnError &err) {
        std::cerr << "Exception: " << err.what() << std::endl;
    }
    return 0;
}
