/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mnn/core/loss/apply_grad.h"
#include "mnn/core/graph/sequential.h"
#include "mnn/core/graph/nodes.h"
#include "mnn/infra/util.h"

namespace mnn {

enum class content_type {
  weights,           ///< save/load the weights
  model,             ///< save/load the network architecture
  weights_and_model  ///< save/load both the weights and the architecture
};

enum class file_format { binary, portable_binary, json };

struct result {
  result() : num_success(0), num_total(0) {}

  float_t accuracy() const { return float_t(num_success * 100.0 / num_total); }

  template <typename Char, typename CharTraits>
  void print_detail(std::basic_ostream<Char, CharTraits> &os) const {
      os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total
         << ")" << std::endl;
  }

  int num_success;
  int num_total;
  std::map<label_t, std::map<label_t, int>> confusion_matrix;
};

template <typename NetType>
class network;

template <typename Layer>
network<sequential> &operator<<(network<sequential> &n, Layer &&l);

template <typename NetType>
class network {
 public:
  typedef typename std::vector<layer *>::iterator iterator;
  typedef typename std::vector<layer *>::const_iterator const_iterator;

  explicit network(const std::string &name = "")
    : name_(name), stop_training_(false) {}

  std::string name() const { return name_; }
  void init_weight() { net_.setup(true); }

  // convenience wrapper for the function below
  template <typename E>
  void bprop(const std::vector<vec_t> &out,
             const std::vector<vec_t> &t,
             const std::vector<vec_t> &t_cost) {
    bprop<E>(std::vector<tensor_t>{out}, std::vector<tensor_t>{t},
             std::vector<tensor_t>{t_cost});
  }

  template <typename E>
  void bprop(const std::vector<tensor_t> &out,
             const std::vector<tensor_t> &t,
             const std::vector<tensor_t> &t_cost) {
    std::vector<tensor_t> delta = gradient<E>(out, t, t_cost);
    nodes& nodes = net_;
    nodes.backward(delta);
  }

  vec_t fprop(const vec_t &in) {
    if (in.size() != (size_t)in_data_size()) data_mismatch(**net_.begin(), in);
    std::vector<tensor_t> a(1);
    a[0].emplace_back(in);
    return fprop(a)[0][0];
  }

  // convenience wrapper for the function below
  std::vector<vec_t> fprop(const std::vector<vec_t> &in) {
    return fprop(std::vector<tensor_t>{in})[0];
  }

  std::vector<tensor_t> fprop(const std::vector<tensor_t> &in) {
      nodes& nodes = net_;
    return nodes.forward(in);
  }

  void update_weights(optimizer *opt) {
    for (auto l : net_) {
      l->update_weight(opt);
    }
  }

  vec_t predict(const vec_t &in) { return fprop(in); }

  tensor_t predict(const tensor_t &in) { return fprop(in); }

  std::vector<tensor_t> predict(const std::vector<tensor_t> &in) {
    return fprop(in);
  }

  float_t predict_max_value(const vec_t &in) { return fprop_max(in); }
  label_t predict_label(const vec_t &in) { return fprop_max_index(in); }

  template <typename Error,
            typename Optimizer,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate>
  bool train(Optimizer &optimizer,
             const std::vector<vec_t> &inputs,
             const std::vector<label_t> &class_labels,
             size_t batch_size,
             int epoch,
             OnBatchEnumerate on_batch_enumerate,
             OnEpochEnumerate on_epoch_enumerate,
             const bool reset_weights         = false,
             const int n_threads              = MNN_TASK_SIZE,
             const std::vector<vec_t> &t_cost = std::vector<vec_t>()) {
    if (inputs.size() != class_labels.size()) {
      return false;
    }
    if (inputs.size() < batch_size || class_labels.size() < batch_size) {
      return false;
    }
    std::vector<tensor_t> input_tensor, output_tensor, t_cost_tensor;
    normalize_tensor(inputs, input_tensor);
    normalize_tensor(class_labels, output_tensor);
    if (!t_cost.empty()) normalize_tensor(t_cost, t_cost_tensor);

    return fit<Error>(optimizer, input_tensor, output_tensor, batch_size, epoch,
                      on_batch_enumerate, on_epoch_enumerate, reset_weights,
                      n_threads, t_cost_tensor);
  }

  template <typename Error,
            typename Optimizer,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate,
            typename T,
            typename U>
  bool fit(Optimizer &optimizer,
           const std::vector<T> &inputs,
           const std::vector<U> &desired_outputs,
           size_t batch_size,
           int epoch,
           OnBatchEnumerate on_batch_enumerate,
           OnEpochEnumerate on_epoch_enumerate,
           const bool reset_weights     = false,
           const int n_threads          = MNN_TASK_SIZE,
           const std::vector<U> &t_cost = std::vector<U>()) {
    std::vector<tensor_t> input_tensor, output_tensor, t_cost_tensor;
    normalize_tensor(inputs, input_tensor);
    normalize_tensor(desired_outputs, output_tensor);
    if (!t_cost.empty()) normalize_tensor(t_cost, t_cost_tensor);

    return fit<Error>(optimizer, input_tensor, output_tensor, batch_size, epoch,
                      on_batch_enumerate, on_epoch_enumerate, reset_weights,
                      n_threads, t_cost_tensor);
  }

  template <typename Error, typename Optimizer>
  bool train(Optimizer &optimizer,
             const std::vector<vec_t> &inputs,
             const std::vector<label_t> &class_labels,
             size_t batch_size = 1,
             int epoch         = 1) {
    return train<Error>(optimizer, inputs, class_labels, batch_size, epoch, nop,
                        nop);
  }

  void set_netphase(net_phase phase) {
    for (auto n : net_) {
      n->set_context(phase);
    }
  }

  void stop_ongoing_training() { stop_training_ = true; }

  result test(const std::vector<vec_t> &in, const std::vector<label_t> &t) {
    result test_result;
    set_netphase(net_phase::test);
    for (size_t i = 0; i < in.size(); i++) {
      const label_t predicted = fprop_max_index(in[i]);
      const label_t actual    = t[i];

      if (predicted == actual) test_result.num_success++;
      test_result.num_total++;
      test_result.confusion_matrix[predicted][actual]++;
    }
    return test_result;
  }

  size_t layer_size() const { return net_.size(); }

  size_t out_data_size() const { return net_.out_data_size(); }
  size_t in_data_size() const { return net_.in_data_size(); }

  template <typename WeightInit>
  network &weight_init(const WeightInit &f) {
    auto ptr = std::make_shared<WeightInit>(f);
    for (auto &l : net_) l->weight_init(ptr);
    return *this;
  }

  template <typename BiasInit>
  network &bias_init(const BiasInit &f) {
    auto ptr = std::make_shared<BiasInit>(f);
    for (auto &l : net_) l->bias_init(ptr);
    return *this;
  }

  iterator begin() { return net_.begin(); }
  iterator end() { return net_.end(); }
  const_iterator begin() const { return net_.begin(); }
  const_iterator end() const { return net_.end(); }

 protected:
  float_t fprop_max(const vec_t &in) {
    const vec_t &prediction = fprop(in);
    return *std::max_element(std::begin(prediction), std::end(prediction));
  }

  label_t fprop_max_index(const vec_t &in) {
    return label_t(max_index(fprop(in)));
  }

 private:
  template <typename Layer>
  friend network<sequential> &operator<<(network<sequential> &n, Layer &&l);

  template <typename Error,
            typename Optimizer,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate>
  bool fit(Optimizer &optimizer,
           const std::vector<tensor_t> &inputs,
           const std::vector<tensor_t> &desired_outputs,
           size_t batch_size,
           int epoch,
           OnBatchEnumerate on_batch_enumerate,
           OnEpochEnumerate on_epoch_enumerate,
           const bool reset_weights            = false,
           const int n_threads                 = MNN_TASK_SIZE,
           const std::vector<tensor_t> &t_cost = std::vector<tensor_t>()) {
    check_target_cost_matrix(desired_outputs, t_cost);
    set_netphase(net_phase::train);
    net_.setup(reset_weights);

    for (auto n : net_) n->set_parallelize(true);
    optimizer.reset();
    stop_training_ = false;
    in_batch_.resize(batch_size);
    t_batch_.resize(batch_size);
    for (int iter = 0; iter < epoch && !stop_training_; iter++) {
      for (size_t i = 0; i < inputs.size() && !stop_training_;
           i += batch_size) {
        train_once<Error>(
          optimizer, &inputs[i], &desired_outputs[i],
          static_cast<int>(std::min(batch_size, (size_t)inputs.size() - i)),
          n_threads, get_target_cost_sample_pointer(t_cost, i));
        on_batch_enumerate();
      }
      on_epoch_enumerate();
    }
    set_netphase(net_phase::test);
    return true;
  }

  template <typename E, typename Optimizer>
  void train_once(Optimizer &optimizer,
                  const tensor_t *in,
                  const tensor_t *t,
                  int size,
                  const int nbThreads,
                  const tensor_t *t_cost) {
    if (size == 1) {
      bprop<E>(fprop(in[0]), t[0], t_cost ? t_cost[0] : tensor_t());
      net_.update_weights(&optimizer);
    } else {
      train_onebatch<E>(optimizer, in, t, size, nbThreads, t_cost);
    }
  }

  template <typename E, typename Optimizer>
  void train_onebatch(Optimizer &optimizer,
                      const tensor_t *in,
                      const tensor_t *t,
                      int batch_size,
                      const int num_tasks,
                      const tensor_t *t_cost) {
    MNN_UNREFERENCED_PARAMETER(num_tasks);
    std::copy(&in[0], &in[0] + batch_size, &in_batch_[0]);
    std::copy(&t[0], &t[0] + batch_size, &t_batch_[0]);
    std::vector<tensor_t> t_cost_batch =
      t_cost ? std::vector<tensor_t>(&t_cost[0], &t_cost[0] + batch_size)
             : std::vector<tensor_t>();

    bprop<E>(fprop(in_batch_), t_batch_, t_cost_batch);
    net_.update_weights(&optimizer);
  }

  void check_target_cost_matrix(const std::vector<tensor_t> &t,
                                const std::vector<tensor_t> &t_cost) {
    if (!t_cost.empty()) {
      if (t.size() != t_cost.size()) {
        throw nn_error(
          "if target cost is supplied, "
          "its length must equal that of target data");
      }

      for (size_t i = 0, end = t.size(); i < end; i++) {
        check_target_cost_element(t[i], t_cost[i]);
      }
    }
  }

  // regression
  void check_target_cost_element(const vec_t &t, const vec_t &t_cost) {
    if (t.size() != t_cost.size()) {
      throw nn_error(
        "if target cost is supplied for a regression task, "
        "its shape must be identical to the target data");
    }
  }

  void check_target_cost_element(const tensor_t &t, const tensor_t &t_cost) {
    if (t.size() != t_cost.size()) {
      throw nn_error(
        "if target cost is supplied for a regression task, "
        "its shape must be identical to the target data");
    }
    for (size_t i = 0; i < t.size(); i++)
      check_target_cost_element(t[i], t_cost[i]);
  }

  const tensor_t *get_target_cost_sample_pointer(
    const std::vector<tensor_t> &t_cost, size_t i) {
    if (!t_cost.empty()) {
      assert(i < t_cost.size());
      return &(t_cost[i]);
    } else {
      return nullptr;
    }
  }

  void normalize_tensor(const std::vector<tensor_t> &inputs,
                        std::vector<tensor_t> &normalized) {
    normalized = inputs;
  }

  void normalize_tensor(const std::vector<vec_t> &inputs,
                        std::vector<tensor_t> &normalized) {
    normalized.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
      normalized.emplace_back(tensor_t{inputs[i]});
  }

  void normalize_tensor(const std::vector<label_t> &inputs,
                        std::vector<tensor_t> &normalized) {
    std::vector<vec_t> vec;
    normalized.reserve(inputs.size());
    net_.label2vec(inputs, vec);
    normalize_tensor(vec, normalized);
  }

  std::string name_;
  NetType net_;
  bool stop_training_;
  std::vector<tensor_t> in_batch_;
  std::vector<tensor_t> t_batch_;
};

template <typename Layer>
network<sequential> &operator<<(network<sequential> &n, Layer &&l) {
  n.net_.add(std::forward<Layer>(l));
  return n;
}
}  // namespace mnn
