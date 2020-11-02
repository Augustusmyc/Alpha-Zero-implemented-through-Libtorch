#pragma once
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <gomoku.h>
#include <common.h>

using namespace torch;
//using namespace customType;



struct AlphaZeroNet : nn::Module{
  public:
    AlphaZeroNet(const unsigned int num_layers, int64_t num_channels, unsigned int n, unsigned int action_size);
    std::pair<Tensor,Tensor> forward(Tensor inputs);
  private:
    nn::Sequential res_layers;
    nn::Conv2d p_conv;
    nn::Conv2d v_conv;
    nn::BatchNorm2d p_bn;
    nn::BatchNorm2d v_bn;
    nn::Linear p_fc;
    nn::Linear v_fc1;
    nn::Linear v_fc2;
    // nn::ReLU relu;
    nn::LogSoftmax p_log_softmax;
};

class NeuralNetwork {
 public:
  using return_type = std::vector<std::vector<double>>;

  NeuralNetwork(std::string model_path, unsigned int batch_size);
#ifndef JIT_MODE
  void save_weights(std::string model_path);
  void load_weights(std::string model_path);
#endif
  ~NeuralNetwork();


  //void train(board_buff_type board_buffer, p_buff_type p_buffer, v_buff_type v_buffer);

  std::future<return_type> commit(Gomoku* gomoku);  // commit task to queue
  //std::shared_ptr<torch::jit::script::Module> module;  // torch module    origin:private
  static Tensor transorm_gomoku_to_Tensor(Gomoku* gomoku);
  static Tensor transorm_board_to_Tensor(board_type board, int last_move, int cur_player);
  unsigned int batch_size;                             // batch size

 private:
  using task_type = std::pair<Tensor, std::promise<return_type>>;
  //optim::Adam *optimizer;
#ifdef JIT_MODE
  std::shared_ptr<torch::jit::script::Module> module;
#else
  std::shared_ptr<AlphaZeroNet> module;
#endif

  void infer();  // infer
  

  std::unique_ptr<std::thread> loop;  // call infer in loop
  bool running;                       // is running

  std::queue<task_type> tasks;  // tasks queue
  std::mutex lock;              // lock for tasks queue
  std::condition_variable cv;   // condition variable for tasks queue

  

};
