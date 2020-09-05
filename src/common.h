#pragma once


#define WHITE -BLACK

// gomoku
#define BORAD_SIZE 3
#define N_IN_ROW 3
#define BLACK 1

// mcts
#define ACTIONSIZE = BORAD_SIZE*BORAD_SIZE
//#define USE_GPU
#define NUM_MCT_THREADS 1
#define NUM_MCT_SIMS 7
#define C_PUCT 5
#define C_VIRTUAL_LOSS 3


// neural_network
#define LR 0.001
#define L2 0.0001
#define NUM_CHANNELS 16 //256
#define NUM_LAYERS 2 //4
#define EPOCHS 1.5
#define BATCH_SIZE 3 //512

#define NUM_TRAIN_THREADS 1


#include <libtorch.h>
using namespace torch;
namespace customType {
    using v_buff_type = std::vector<int>;
    using p_buff_type = std::vector<std::vector<double>>;
    using board_buff_type = std::vector<Tensor>;
}
