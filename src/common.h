#pragma once

#define DEBUG_MODE



#ifdef DEBUG_MODE
    #define BORAD_SIZE 3
    #define N_IN_ROW 3
    #define NUM_MCT_THREADS 4
    #define NUM_MCT_SIMS 1600
    #define EXPLORE_STEP 5
    #define C_PUCT 3
    #define C_VIRTUAL_LOSS 1
    #define NUM_CHANNELS 256
    #define NUM_LAYERS 4

    #define BATCH_SIZE 256 //512
    #define DIRI 0.2

    #define NUM_TRAIN_THREADS 4
#else
    #define BORAD_SIZE 11
    #define N_IN_ROW 5
    #define NUM_MCT_THREADS 4
    #define NUM_MCT_SIMS 1600
    #define EXPLORE_STEP 5
    #define C_PUCT 5
    #define C_VIRTUAL_LOSS 3
    #define NUM_CHANNELS 256
    #define NUM_LAYERS 4

    #define BATCH_SIZE 256
    #define DIRI 0.1

    #define NUM_TRAIN_THREADS 20
#endif

#define BLACK 1
#define WHITE -BLACK

#define USE_GPU

#define LR 0.001
//#define L2 0.0001

#define BUFFER_LEN BORAD_SIZE*BORAD_SIZE+1

namespace customType {
    using v_buff_type = std::vector<int>;
    using p_buff_type = std::vector<std::vector<float>>;
    using board_type = std::vector<std::vector<int>>;
    using board_buff_type = std::vector<board_type>;;
}
