#pragma once

#define BORAD_SIZE 11
#define N_IN_ROW 5
#define BLACK 1
#define WHITE -BLACK
#define USE_GPU true

#include <libtorch.h>
using namespace torch;
namespace customType {
    using v_buff_type = std::vector<int>;
    using p_buff_type = std::vector<std::vector<double>>;
    using board_buff_type = std::vector<Tensor>;
}
