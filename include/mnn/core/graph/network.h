/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include "mnn/core/graph/node_list.h"
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
#include "mnn/infra/util.h"
#include "mnn/core/graph/sequential.h"

namespace mnn {

enum class ContentType {
  weights,           ///< save/load the weights
  model,             ///< save/load the network architecture
  weights_and_model  ///< save/load both the weights and the architecture
};

struct Result {
  Result() : num_success(0), num_total(0) {}

  Float accuracy() const { return Float(num_success * 100.0 / num_total); }

  template <typename Char, typename CharTraits>
  void print_detail(std::basic_ostream<Char, CharTraits> &os) const {
      os << "accuracy:" << accuracy() << "% (" << num_success << "/" << num_total
         << ")" << std::endl;
  }

  int num_success;
  int num_total;
  std::map<Label, std::map<Label, int>> confusion_matrix;
};

template <typename NetType>
class Network : NetType {
 public:
  typedef typename std::vector<Layer *>::iterator iterator;
  typedef typename std::vector<Layer *>::const_iterator const_iterator;

  explicit Network(const std::string &name = "")
    : name_(name), stop_training_(false) {}

  std::string name() const { return name_; }
  void init_weight() { NetType::setup(true); }

  template <typename Layer>
  void add(Layer &&layer) {
      NetType::add(std::forward<Layer>(layer));
  }

  // convenience wrapper for the function below
  template <typename E>
  void bprop(const std::vector<Vector> &out,
             const std::vector<Vector> &t,
             const std::vector<Vector> &t_cost) {
    bprop<E>(std::vector<Matrix>{out}, std::vector<Matrix>{t},
             std::vector<Matrix>{t_cost});
  }

  template <typename E>
  void bprop(const std::vector<Matrix> &out,
             const std::vector<Matrix> &t,
             const std::vector<Matrix> &t_cost) {
    std::vector<Matrix> delta = gradient<E>(out, t, t_cost);
    NodeList& nodes = *this;
    nodes.backward(delta);
  }

  Vector fprop(const Vector &in) {
    if (in.size() != (size_t)in_data_size()) data_mismatch(**NetType::begin(), in);
    std::vector<Matrix> a(1);
    a[0].emplace_back(in);
    return fprop(a)[0][0];
  }

  // convenience wrapper for the function below
  std::vector<Vector> fprop(const std::vector<Vector> &in) {
    return fprop(std::vector<Matrix>{in})[0];
  }

  std::vector<Matrix> fprop(const std::vector<Matrix> &in) {
      NodeList& nodes = *this;
    return nodes.forward(in);
  }

  void update_weights(Optimizer *opt) {
    for (auto l : *this) {
      l->update_weight(opt);
    }
  }

  Vector predict(const Vector &in) { return fprop(in); }

  Matrix predict(const Matrix &in) { return fprop(in); }

  std::vector<Matrix> predict(const std::vector<Matrix> &in) {
    return fprop(in);
  }

  Float predict_max_value(const Vector &in) { return fprop_max(in); }
  Label predict_label(const Vector &in) { return fprop_max_index(in); }

  template <typename Error,
            typename Optimizer,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate>
  bool train(Optimizer &optimizer,
             const std::vector<Vector> &inputs,
             const std::vector<Label> &class_labels,
             size_t batch_size,
             int epoch,
             OnBatchEnumerate on_batch_enumerate,
             OnEpochEnumerate on_epoch_enumerate,
             const bool reset_weights         = false,
             const int n_threads              = MNN_TASK_SIZE,
             const std::vector<Vector> &t_cost = std::vector<Vector>()) {
    if (inputs.size() != class_labels.size()) {
      return false;
    }
    if (inputs.size() < batch_size || class_labels.size() < batch_size) {
      return false;
    }
    std::vector<Matrix> input_tensor, output_tensor, t_cost_tensor;
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
    std::vector<Matrix> input_tensor, output_tensor, t_cost_tensor;
    normalize_tensor(inputs, input_tensor);
    normalize_tensor(desired_outputs, output_tensor);
    if (!t_cost.empty()) normalize_tensor(t_cost, t_cost_tensor);

    return fit<Error>(optimizer, input_tensor, output_tensor, batch_size, epoch,
                      on_batch_enumerate, on_epoch_enumerate, reset_weights,
                      n_threads, t_cost_tensor);
  }

  template <typename Error, typename Optimizer>
  bool train(Optimizer &optimizer,
             const std::vector<Vector> &inputs,
             const std::vector<Label> &class_labels,
             size_t batch_size = 1,
             int epoch         = 1) {
    return train<Error>(optimizer, inputs, class_labels, batch_size, epoch, nop,
                        nop);
  }

  void set_netphase(NetPhase phase) {
    for (auto n : *this) {
      n->set_context(phase);
    }
  }

  void stop_ongoing_training() { stop_training_ = true; }

  Result test(const std::vector<Vector> &in, const std::vector<Label> &t) {
    Result test_result;
    set_netphase(NetPhase::TESTING);
    for (size_t i = 0; i < in.size(); i++) {
      const Label predicted = fprop_max_index(in[i]);
      const Label actual    = t[i];

      if (predicted == actual) test_result.num_success++;
      test_result.num_total++;
      test_result.confusion_matrix[predicted][actual]++;
    }
    return test_result;
  }

  size_t layer_size() const { return NetType::size(); }

  size_t out_data_size() const { return NetType::out_data_size(); }
  size_t in_data_size() const { return NetType::in_data_size(); }

  template <typename WeightInit>
  Network &weight_init(const WeightInit &f) {
    auto ptr = std::make_shared<WeightInit>(f);
    for (auto &l : *this) l->weight_init(ptr);
    return *this;
  }

  template <typename BiasInit>
  Network &bias_init(const BiasInit &f) {
    auto ptr = std::make_shared<BiasInit>(f);
    for (auto &l : *this) l->bias_init(ptr);
    return *this;
  }

  iterator begin() { return NetType::begin(); }
  iterator end() { return NetType::end(); }
  const_iterator begin() const { return NetType::begin(); }
  const_iterator end() const { return NetType::end(); }

 protected:
  Float fprop_max(const Vector &in) {
    const Vector &prediction = fprop(in);
    return *std::max_element(std::begin(prediction), std::end(prediction));
  }

  Label fprop_max_index(const Vector &in) {
    return Label(max_index(fprop(in)));
  }

 private:
  template <typename Layer>
  friend Network<Sequential> &operator<<(Network<Sequential> &n, Layer &&l);

  template <typename Error,
            typename Optimizer,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate>
  bool fit(Optimizer &optimizer,
           const std::vector<Matrix> &inputs,
           const std::vector<Matrix> &desired_outputs,
           size_t batch_size,
           int epoch,
           OnBatchEnumerate on_batch_enumerate,
           OnEpochEnumerate on_epoch_enumerate,
           const bool reset_weights            = false,
           const int n_threads                 = MNN_TASK_SIZE,
           const std::vector<Matrix> &t_cost = std::vector<Matrix>()) {
    check_target_cost_matrix(desired_outputs, t_cost);
    set_netphase(NetPhase::TRAINING);
    NetType::setup(reset_weights);

    for (auto n : *this) n->set_parallelize(true);
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
    set_netphase(NetPhase::TESTING);
    return true;
  }

  template <typename E, typename Optimizer>
  void train_once(Optimizer &optimizer,
                  const Matrix *in,
                  const Matrix *t,
                  int size,
                  const int nbThreads,
                  const Matrix *t_cost) {
    if (size == 1) {
      bprop<E>(fprop(in[0]), t[0], t_cost ? t_cost[0] : Matrix());
      NetType::update_weights(&optimizer);
    } else {
      train_onebatch<E>(optimizer, in, t, size, nbThreads, t_cost);
    }
  }

  template <typename E, typename Optimizer>
  void train_onebatch(Optimizer &optimizer,
                      const Matrix *in,
                      const Matrix *t,
                      int batch_size,
                      const int num_tasks,
                      const Matrix *t_cost) {
    MNN_UNREFERENCED_PARAMETER(num_tasks);
    std::copy(&in[0], &in[0] + batch_size, &in_batch_[0]);
    std::copy(&t[0], &t[0] + batch_size, &t_batch_[0]);
    std::vector<Matrix> t_cost_batch =
      t_cost ? std::vector<Matrix>(&t_cost[0], &t_cost[0] + batch_size)
             : std::vector<Matrix>();

    bprop<E>(fprop(in_batch_), t_batch_, t_cost_batch);
    NetType::update_weights(&optimizer);
  }

  void check_target_cost_matrix(const std::vector<Matrix> &t,
                                const std::vector<Matrix> &t_cost) {
    if (!t_cost.empty()) {
      if (t.size() != t_cost.size()) {
        throw MnnError(
          "if target cost is supplied, "
          "its length must equal that of target data");
      }

      for (size_t i = 0, end = t.size(); i < end; i++) {
        check_target_cost_element(t[i], t_cost[i]);
      }
    }
  }

  // regression
  void check_target_cost_element(const Vector &t, const Vector &t_cost) {
    if (t.size() != t_cost.size()) {
      throw MnnError(
        "if target cost is supplied for a regression task, "
        "its shape must be identical to the target data");
    }
  }

  void check_target_cost_element(const Matrix &t, const Matrix &t_cost) {
    if (t.size() != t_cost.size()) {
      throw MnnError(
        "if target cost is supplied for a regression task, "
        "its shape must be identical to the target data");
    }
    for (size_t i = 0; i < t.size(); i++)
      check_target_cost_element(t[i], t_cost[i]);
  }

  const Matrix *get_target_cost_sample_pointer(
    const std::vector<Matrix> &t_cost, size_t i) {
    if (!t_cost.empty()) {
      assert(i < t_cost.size());
      return &(t_cost[i]);
    } else {
      return nullptr;
    }
  }

  void normalize_tensor(const std::vector<Matrix> &inputs,
                        std::vector<Matrix> &normalized) {
    normalized = inputs;
  }

  void normalize_tensor(const std::vector<Vector> &inputs,
                        std::vector<Matrix> &normalized) {
    normalized.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
      normalized.emplace_back(Matrix{inputs[i]});
  }

  void normalize_tensor(const std::vector<Label> &inputs,
                        std::vector<Matrix> &normalized) {
    std::vector<Vector> vec;
    normalized.reserve(inputs.size());
    NetType::label2vec(inputs, vec);
    normalize_tensor(vec, normalized);
  }

  std::string name_;
  bool stop_training_;
  std::vector<Matrix> in_batch_;
  std::vector<Matrix> t_batch_;
};
}  // namespace mnn
