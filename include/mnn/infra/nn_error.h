/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once

#include <exception>
#include <string>

namespace mnn {

class nn_error : public std::exception {
 public:
  explicit nn_error(const std::string &msg) : msg_(msg) {}
  const char *what() const throw() override { return msg_.c_str(); }

 private:
  std::string msg_;
};

}  // namespace mnn
