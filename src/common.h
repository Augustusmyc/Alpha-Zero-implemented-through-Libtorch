#pragma once


#define WHITE -BLACK


// gomoku
#define BORAD_SIZE 11
#define N_IN_ROW 5
#define BLACK 1

// mcts
#define ACTIONSIZE = BORAD_SIZE*BORAD_SIZE
#define USE_GPU true
#define NUM_MCT_THREADS 4
#define NUM_MCT_SIMS 1600
#define C_PUCT 5
#define C_VIRTUAL_LOSS 3


// neural_network
#define LR 0.001
#define L2 0.0001
#define NUM_CHANNELS 256
#define NUM_LAYERS 4
#define EPOCHS 1.5
#define BATCH_SIZE 32 //512

#define NUM_TRAIN_THREADS 10


#include <libtorch.h>
using namespace torch;
namespace customType {
    using v_buff_type = std::vector<int>;
    using p_buff_type = std::vector<std::vector<double>>;
    using board_buff_type = std::vector<Tensor>;
}
