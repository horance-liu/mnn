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

enum class BackendType;

class Layer: public Node {
public:
    friend void connection_mismatch(const Layer &from, const Layer &to);

    virtual ~Layer() = default;

    Layer(const std::vector<VectorType> &in_type,
            const std::vector<VectorType> &out_type);

    Layer(const Layer&) = default;
    Layer& operator=(const Layer&) = default;

    Layer(Layer&&) = default;
    Layer& operator=(Layer&&) = default;

    void set_parallelize(bool parallelize);
    void set_backend_type(BackendType backend_type);

    bool parallelize() const;
    BackendType engine() const;

    size_t in_channels() const;
    size_t out_channels() const;

    size_t in_data_size() const;
    size_t out_data_size() const;

    std::vector<Shape3d> in_data_shape();

    std::vector<Shape3d> out_data_shape();

    std::vector<const Vector*> weights() const;

    std::vector<Vector*> weights();

    std::vector<Matrix*> weights_grads();

    std::vector<edgeptr_t> inputs();
    std::vector<edgeptr_t> outputs();

    std::vector<edgeptr_t> outputs() const;

    void set_out_grads(const std::vector<const Vector*> *grad, size_t cnt);
    void set_in_data(const std::vector<const Vector*> *data, size_t cnt);

    void output(std::vector<const Matrix*> &out) const;

    std::vector<VectorType> in_types() const;
    std::vector<VectorType> out_types() const;

    void set_trainable(bool trainable);
    bool trainable() const;

    virtual std::vector<Shape3d> in_shape() const = 0;
    virtual std::vector<Shape3d> out_shape() const = 0;
    virtual std::string layer_type() const = 0;

    virtual std::pair<Float, Float> out_value_range() const;
    virtual void set_in_shape(const Shape3d &in_shape);
    virtual size_t fan_in_size() const;
    virtual size_t fan_in_size(size_t) const;
    virtual size_t fan_out_size() const;
    virtual size_t fan_out_size(size_t) const;
    virtual void set_sample_count(size_t sample_count);

    template<typename WeightInit>
    Layer& weight_init(const WeightInit &f)
    {
        weight_init_ = std::make_shared<WeightInit>(f);
        return *this;
    }

    template<typename BiasInit>
    Layer& bias_init(const BiasInit &f)
    {
        bias_init_ = std::make_shared<BiasInit>(f);
        return *this;
    }

    template<typename WeightInit>
    Layer& weight_init(std::shared_ptr<WeightInit> f)
    {
        weight_init_ = f;
        return *this;
    }

    template<typename BiasInit>
    Layer& bias_init(std::shared_ptr<BiasInit> f)
    {
        bias_init_ = f;
        return *this;
    }

    virtual void forward_propagation(
            const std::vector<Matrix*> &in_data,
            std::vector<Matrix*> &out_data) = 0;

    virtual void back_propagation(
            const std::vector<Matrix*> &in_data,
            const std::vector<Matrix*> &out_data,
            std::vector<Matrix*> &out_grad,
            std::vector<Matrix*> &in_grad) = 0;

    virtual void post_update() {}
    virtual void set_context(NetPhase) {}

    std::vector<Matrix> backward(const std::vector<Matrix> &out_grads);

    void forward();
    void backward();

    void setup(bool reset_weight);

    void init_weight();
    void clear_grads();

    void update_weight(Optimizer *o);
    bool has_same_weights(const Layer &rhs, Float eps) const;

protected:
    bool initialized_;
    bool parallelize_;

    size_t in_channels_;
    size_t out_channels_;

    std::vector<VectorType> in_type_;
    std::vector<VectorType> out_type_;

    BackendType backend_type_;
    Vector weights_diff_;

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

    Vector* get_weight_data(size_t i);
    const Vector* get_weight_data(size_t i) const;

private:
    bool trainable_;
    std::shared_ptr<weight_init::Function> weight_init_;
    std::shared_ptr<weight_init::Function> bias_init_;

    std::vector<Matrix*> fwd_in_data_;
    std::vector<Matrix*> fwd_out_data_;
    std::vector<Matrix*> bwd_in_data_;
    std::vector<Matrix*> bwd_in_grad_;
    std::vector<Matrix*> bwd_out_data_;
    std::vector<Matrix*> bwd_out_grad_;
};

Layer& operator<<(Layer &lhs, Layer &rhs);

void data_mismatch(const Layer &layer, const Vector &data);
void pooling_size_mismatch(size_t in_width, size_t in_height,
        size_t pooling_size_x, size_t pooling_size_y);

}  // namespace mnn
