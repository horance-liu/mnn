/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/node.h"
#include "mnn/core/optimizer/optimizer.h"
#include "mnn/infra/weight_init.h"

namespace mnn {

enum class backend_t;

class layer: public node {
public:
    friend void connection_mismatch(const layer &from, const layer &to);

    virtual ~layer() = default;

    layer(const std::vector<vector_type> &in_type,
            const std::vector<vector_type> &out_type);

    layer(const layer&) = default;
    layer& operator=(const layer&) = default;

    layer(layer&&) = default;
    layer& operator=(layer&&) = default;

    void set_parallelize(bool parallelize);
    void set_backend_type(backend_t backend_type);

    bool parallelize() const;
    backend_t engine() const;

    size_t in_channels() const;
    size_t out_channels() const;

    size_t in_data_size() const;
    size_t out_data_size() const;

    std::vector<shape3d> in_data_shape();

    std::vector<shape3d> out_data_shape();

    std::vector<const vec_t*> weights() const;

    std::vector<vec_t*> weights();

    std::vector<tensor_t*> weights_grads();

    std::vector<edgeptr_t> inputs();
    std::vector<edgeptr_t> outputs();

    std::vector<edgeptr_t> outputs() const;

    void set_out_grads(const std::vector<const vec_t*> *grad, size_t cnt);
    void set_in_data(const std::vector<const vec_t*> *data, size_t cnt);

    void output(std::vector<const tensor_t*> &out) const;

    std::vector<vector_type> in_types() const;
    std::vector<vector_type> out_types() const;

    void set_trainable(bool trainable);
    bool trainable() const;

    virtual std::vector<shape3d> in_shape() const = 0;
    virtual std::vector<shape3d> out_shape() const = 0;
    virtual std::string layer_type() const = 0;

    virtual std::pair<float_t, float_t> out_value_range() const;
    virtual void set_in_shape(const shape3d &in_shape);
    virtual size_t fan_in_size() const;
    virtual size_t fan_in_size(size_t) const;
    virtual size_t fan_out_size() const;
    virtual size_t fan_out_size(size_t) const;
    virtual void set_sample_count(size_t sample_count);

    template<typename WeightInit>
    layer& weight_init(const WeightInit &f)
    {
        weight_init_ = std::make_shared<WeightInit>(f);
        return *this;
    }

    template<typename BiasInit>
    layer& bias_init(const BiasInit &f)
    {
        bias_init_ = std::make_shared<BiasInit>(f);
        return *this;
    }

    template<typename WeightInit>
    layer& weight_init(std::shared_ptr<WeightInit> f)
    {
        weight_init_ = f;
        return *this;
    }

    template<typename BiasInit>
    layer& bias_init(std::shared_ptr<BiasInit> f)
    {
        bias_init_ = f;
        return *this;
    }

    virtual void forward_propagation(
            const std::vector<tensor_t*> &in_data,
            std::vector<tensor_t*> &out_data) = 0;

    virtual void back_propagation(
            const std::vector<tensor_t*> &in_data,
            const std::vector<tensor_t*> &out_data,
            std::vector<tensor_t*> &out_grad,
            std::vector<tensor_t*> &in_grad) = 0;

    virtual void post_update() {}
    virtual void set_context(net_phase) {}

    std::vector<tensor_t> backward(const std::vector<tensor_t> &out_grads);

    void forward();
    void backward();

    void setup(bool reset_weight);

    void init_weight();
    void clear_grads();

    void update_weight(optimizer *o);
    bool has_same_weights(const layer &rhs, float_t eps) const;

protected:
    bool initialized_;
    bool parallelize_;

    size_t in_channels_;
    size_t out_channels_;

    std::vector<vector_type> in_type_;
    std::vector<vector_type> out_type_;

    backend_t backend_type_;
    vec_t weights_diff_;

    template<typename T, typename Func>
    inline void for_i(T size, Func f, size_t grainsize = 100)
    {
        mnn::for_i(parallelize_, size, f, grainsize);
    }

private:
    void alloc_input(size_t i) const;
    void alloc_output(size_t i) const;

    edgeptr_t ith_in_node(size_t i);
    edgeptr_t ith_out_node(size_t i);
    edgeptr_t ith_out_node(size_t i) const;

    vec_t* get_weight_data(size_t i);
    const vec_t* get_weight_data(size_t i) const;

private:
    bool trainable_;
    std::shared_ptr<weight_init::function> weight_init_;
    std::shared_ptr<weight_init::function> bias_init_;

    std::vector<tensor_t*> fwd_in_data_;
    std::vector<tensor_t*> fwd_out_data_;
    std::vector<tensor_t*> bwd_in_data_;
    std::vector<tensor_t*> bwd_in_grad_;
    std::vector<tensor_t*> bwd_out_data_;
    std::vector<tensor_t*> bwd_out_grad_;
};

layer& operator<<(layer &lhs, layer &rhs);

void data_mismatch(const layer &layer, const vec_t &data);
void pooling_size_mismatch(size_t in_width, size_t in_height,
        size_t pooling_size_x, size_t pooling_size_y);

}  // namespace mnn
