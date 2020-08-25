#pragma once

//#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <gomoku.h>

using namespace torch;

static Tensor transorm_gomoku_to_Tensor(Gomoku* gomoku);

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

  NeuralNetwork(bool use_gpu, unsigned int batch_size);
  ~NeuralNetwork();

  void save_weights(string model_path);
  void load_weights(string model_path);
  void train();

  std::future<return_type> commit(Gomoku* gomoku);  // commit task to queue
  //std::shared_ptr<torch::jit::script::Module> module;  // torch module    origin:private
  void set_batch_size(unsigned int batch_size) {    // set batch_size
    this->batch_size = batch_size;
  };
  //torch::optim::SGD optimizer;

 private:
  using task_type = std::pair<Tensor, std::promise<return_type>>;
  optim::Adam optimizer;
  std::shared_ptr<AlphaZeroNet> module;
  

  void infer();  // infer
  std::unique_ptr<std::thread> loop;  // call infer in loop
  bool running;                       // is running

  std::queue<task_type> tasks;  // tasks queue
  std::mutex lock;              // lock for tasks queue
  std::condition_variable cv;   // condition variable for tasks queue

  
  unsigned int batch_size;                             // batch size
  bool use_gpu;                                        // use gpu
};
