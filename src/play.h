#pragma once

#include <torch/script.h>

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <libtorch.h>
#include <thread_pool.h>
#include <gomoku.h>
#include <common.h>


class SelfPlay{
    public:
        SelfPlay(NeuralNetwork *nn);
        //~SelfPlay();
        CustomType::p_buff_type self_play_for_train(unsigned int game_num);
        CustomType::board_buff_type get_buffer();
    private:
        void play();
        CustomType::p_buff_type *p_buffer;
        CustomType::board_buff_type *board_buffer;
        NeuralNetwork *nn;
        std::unique_ptr<ThreadPool> thread_pool;
        //std::queue<task_type> tasks;  // tasks queue
        std::mutex lock;              // lock for tasks queue
        std::condition_variable cv;   // condition variable for tasks queue
};