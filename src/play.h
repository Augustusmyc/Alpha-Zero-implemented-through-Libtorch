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
        std::pair<int,int> self_play_for_eval(NeuralNetwork *a, NeuralNetwork *b);
        void play(unsigned int saved_id);

        int play(NeuralNetwork* a, NeuralNetwork* b);

        void self_play_for_train(unsigned int game_num, unsigned int start_batch_id);

        pair<int, int> self_play_for_eval(unsigned int game_num, NeuralNetwork* a, NeuralNetwork* b);
        
    private:
        //p_buff_type *p_buffer;
        //board_buff_type *board_buffer;
        //v_buff_type *v_buffer;
        
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
