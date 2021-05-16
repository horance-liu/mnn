/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/layer/layer.h"
#include <vector>

namespace mnn {

class nodes {
public:
    typedef std::vector<layer*>::iterator iterator;
    typedef std::vector<layer*>::const_iterator const_iterator;

    virtual ~nodes() {}

    virtual void backward(const std::vector<tensor_t> &first) = 0;
    virtual std::vector<tensor_t> forward(const std::vector<tensor_t> &first) = 0;

    virtual void update_weights(optimizer *opt);
    virtual void setup(bool reset_weight);

    void clear_grads();
    size_t size() const;

    iterator begin();
    iterator end();

    const_iterator begin() const;
    const_iterator end() const;

    size_t in_data_size() const;
    size_t out_data_size() const;

    virtual float_t target_value_min(int out_channel = 0) const;
    virtual float_t target_value_max(int out_channel = 0) const;

    void label2vec(const label_t *t, size_t num, std::vector<vec_t> &vec) const;
    void label2vec(const std::vector<label_t> &labels, std::vector<vec_t> &vec) const;

protected:
    template<typename T>
    void push_back(T &&node)
    {
        push_back_impl(
                std::forward<T>(node),
                typename std::is_rvalue_reference<decltype(node)>::type()); // NOLINT
    }

    template<typename T>
    void push_back(std::shared_ptr<T> node)
    {
        own_nodes_.push_back(node);
        nodes_.push_back(own_nodes_.back().get());
    }

    void reorder_for_layerwise_processing(
            const std::vector<tensor_t> &input,
            std::vector<std::vector<const vec_t*>> &output);

    template<typename T>
    void push_back_impl(T &&node, std::true_type)
    {  // is_rvalue_reference
        own_nodes_.push_back(
                std::make_shared<typename std::remove_reference<T>::type>(
                        std::forward<T>(node)));
        nodes_.push_back(own_nodes_.back().get());
    }

    template<typename T>
    void push_back_impl(T &&node, std::false_type)
    {
        nodes_.push_back(&node);
    }

    std::vector<std::shared_ptr<layer>> own_nodes_;
    std::vector<layer*> nodes_;
};

}  // namespace mnn
