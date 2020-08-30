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

using namespace customType;

class SelfPlay{
    public:
        SelfPlay(NeuralNetwork *nn);
        //~SelfPlay();
        std::tuple<board_buff_type, p_buff_type, v_buff_type> self_play_for_train(unsigned int game_num);
        std::pair<int,int> self_play_for_eval(NeuralNetwork *a, NeuralNetwork *b);
        
    private:
        p_buff_type *p_buffer;
        board_buff_type *board_buffer;
        v_buff_type *v_buffer;
        void play();
        NeuralNetwork *nn;
        std::unique_ptr<ThreadPool> thread_pool;
        //std::queue<task_type> tasks;  // tasks queue
        std::mutex lock;              // lock for tasks queue
        std::condition_variable cv;   // condition variable for tasks queue
};

//class EvalPlay {
//    public:
//        EvalPlay(NeuralNetwork* nn);
//        int self_play_for_eval(int first_weight, int second_weight);
//    private:
//        NeuralNetwork* nn;
//        std::unique_ptr<ThreadPool> thread_pool;
//};
