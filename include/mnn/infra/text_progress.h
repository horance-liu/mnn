/*
 *   Copyright (c) 2021, Horance Liu and the respective contributors
 *   All rights reserved.
 *
 *   Use of this source code is governed by a Apache 2.0 license that can be found
 *   in the LICENSE file.
 */

#pragma once
#include <iostream>  // for ostream, cout, etc
#include <string>    // for string

namespace mnn {

class TextProgress {
public:
    explicit TextProgress(size_t expected_count_, std::ostream &os =
            std::cout, const std::string &s1 = "\n",  // leading strings
            const std::string &s2 = "", const std::string &s3 = "");

    void restart(size_t expected_count_);

    size_t operator+=(size_t increment);
    size_t operator++();

    size_t count() const;
    size_t expected_count() const;

private:
    void display_tic();

private:
    std::ostream &m_os;      // may not be present in all imps
    const std::string m_s1;  // string is more general, safer than
    const std::string m_s2;  //  const char *, and efficiency or size are
    const std::string m_s3;  //  not issues

    size_t _count, _expected_count, _next_tic_count;
    size_t _tic;
};

}  // namespace mnn
