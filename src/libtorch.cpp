#include <libtorch.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAGuard.h>

#include <torch/torch.h>
#include <iostream>
#include <common.h>
#include <random>

using namespace std::chrono_literals;
using namespace torch;
using namespace customType;

nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, unsigned int stride=1){
  return nn::Conv2d(nn::Conv2dOptions(/*in_channel=*/in_channels, out_channels, /*kernel_size=*/3).stride(stride).padding(1).bias(false));
}

static inline Tensor alpha_loss(Tensor log_ps, Tensor vs, const Tensor target_ps, const Tensor target_vs){
  return mean(pow(vs - target_vs, 2)) - mean(sum(target_ps * log_ps, 1)); // value_loss + policy_loss
}

struct ResidualBlock:nn::Module {
  ResidualBlock(){
    //only for initializing array
  }

  ResidualBlock(int64_t in_channels, int64_t out_channels, unsigned int stride=1){
    conv1 = register_module("conv1",conv3x3(in_channels, out_channels, stride));
    bn1 = register_module("bn1", nn::BatchNorm2d(out_channels));
    conv2 = register_module("conv2",conv3x3(out_channels, out_channels));
    bn2 = register_module("bn2", nn::BatchNorm2d(out_channels));
    // relu = register_module("relu", nn::ReLU(true));
    this->out_channels = out_channels;
    downsample = false;
    if (in_channels != out_channels || stride != 1){
      downsample = true;
      downsample_conv = register_module("downsample_conv",conv3x3(in_channels, out_channels, stride=stride));
      downsample_bn = register_module("downsample_bn", nn::BatchNorm2d(out_channels));
    }
  }

  Tensor forward(Tensor x) {
    // Use one of many tensor manipulation functions.
    Tensor residual = x;
    Tensor out = relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    if (downsample){
      residual = downsample_conv(residual);
      residual = downsample_bn((residual));
    }

    out += residual;
    out = relu(out);
    return out;
  }

  // Use one of many "standard library" modules.
  nn::Conv2d conv1{nullptr};
  nn::Conv2d conv2{nullptr};
  nn::BatchNorm2d bn1{nullptr};
  nn::BatchNorm2d bn2{nullptr};
  nn::BatchNorm2d downsample_bn{nullptr};
  nn::Conv2d downsample_conv{nullptr};
  // nn::ReLU relu{nullptr};
  bool downsample;
  int64_t out_channels;
};


AlphaZeroNet::AlphaZeroNet(const unsigned int num_layers, int64_t num_channels, unsigned int n, unsigned int action_size)
      :res_layers(nullptr),p_conv(nullptr),v_conv(nullptr),
      p_bn(nullptr),v_bn(nullptr),p_fc(nullptr),v_fc1(nullptr),v_fc2(nullptr), //relu(nullptr),
      p_log_softmax(nullptr){
    // residual block
    ResidualBlock *res_list = new ResidualBlock[num_layers]();//  ResidualBlock(3, num_channels);
    res_list[0] = ResidualBlock(3, num_channels);
    for (unsigned int i = 1; i<num_layers; i++){
        res_list[i] = ResidualBlock(num_channels, num_channels);
     }
    if (USE_GPU) {
        for (unsigned int i = 0; i < num_layers; i++) {
            res_list[i].to(torch::kCUDA);
        }
    }
    res_layers = nn::Sequential(*res_list);

    //  policy head
    p_conv = register_module("p_conv", nn::Conv2d(
      nn::Conv2dOptions(/*in_channel=*/num_channels, 4, /*kernel_size=*/1).padding(0).bias(false)));
    p_bn = register_module("p_bn",nn::BatchNorm2d(4));
    // relu = register_module("relu", nn::ReLU(true));
    p_fc = register_module("p_fc",nn::Linear(4 * n * n, action_size));
    p_log_softmax = register_module("p_log_softmax", nn::LogSoftmax(/*dim=*/1));
    

     // value head
    v_conv = register_module("v_conv",nn::Conv2d(nn::Conv2dOptions(/*in_channel=*/num_channels, 2, /*kernel_size=*/1).padding(0).bias(false)));
    v_bn = register_module("v_bn",nn::BatchNorm2d(2));

    v_fc1 = register_module("v_fc1", nn::Linear(2 * n * n, 256));
    v_fc2 = register_module("v_fc2", nn::Linear(256, 1));
  }

  // Implement the Net's algorithm.
 std::pair<Tensor,Tensor>  AlphaZeroNet::forward(Tensor inputs) {

    Tensor out = res_layers->forward(inputs);

    // policy head
    Tensor p = p_conv(out);
    p = p_bn(p);
    p = relu(p);

    p = p_fc(p.view({p.size(0), -1}));
    p = p_log_softmax(p);

    // value head
    Tensor v = v_conv(out);
    v = v_bn(v);
    v = relu(v);

    v = v_fc1(v.view({v.size(0), -1}));
    v = relu(v);
    v = tanh(v_fc2(v));

    return std::make_pair(p,v);
  }

NeuralNetwork::NeuralNetwork(unsigned int batch_size)
    : //module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
      batch_size(batch_size),
      running(true),
      //optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4)),
      loop(nullptr) {
  module = std::make_shared<AlphaZeroNet>(
  AlphaZeroNet(/*num_layers=*/4,/*num_channels=*/256,/*n=*/BORAD_SIZE,/*action_size=*/BORAD_SIZE * BORAD_SIZE));

  optimizer = new torch::optim::Adam(this->module->parameters(), torch::optim::AdamOptions(2e-4));

  if (USE_GPU) {
    // move to CUDA
    this->module->to(at::kCUDA);
  }

   //this->optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
   //auto optimizer = torch::optim::Adam(this->module->parameters(), 0.01);

  // run infer thread
  this->loop = std::make_unique<std::thread>([this] {
    while (this->running) {
      this->infer();
    }
  });
}

void NeuralNetwork::save_weights(string model_path) {
  torch::save(this->module, model_path.c_str());
}

void NeuralNetwork::load_weights(string model_path) {
  torch::load(this->module, model_path.c_str());
}

NeuralNetwork::~NeuralNetwork() {
  this->running = false;
  this->loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(Gomoku* gomoku) {
  auto states = transorm_gomoku_to_Tensor(gomoku);

  // emplace task
  std::promise<return_type> promise;
  auto ret = promise.get_future();

  {
    std::lock_guard<std::mutex> lock(this->lock);
    tasks.emplace(std::make_pair(states, std::move(promise)));
  }

  this->cv.notify_all();

  return ret;
}

Tensor NeuralNetwork::transorm_gomoku_to_Tensor(Gomoku* gomoku){
  int n = gomoku->get_n();

  // convert data format
  auto board = gomoku->get_board();
  std::vector<int> board0;
  for (unsigned int i = 0; i < board.size(); i++) {
    board0.insert(board0.end(), board[i].begin(), board[i].end());
  }

  torch::Tensor temp =
      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt32));

  torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
  torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

  int last_move = gomoku->get_last_move();
  int cur_player = gomoku->get_current_color();

  if (cur_player == -1) {
    std::swap(state0, state1);
  }

  torch::Tensor state2 =
      torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));

  if (last_move != -1) {
    state2[0][0][last_move / n][last_move % n] = 1;
  }

  // torch::Tensor states = torch::cat({state0, state1}, 1);
  return torch::cat({state0, state1, state2}, 1);
}

void NeuralNetwork::infer() {
    {
        //torch::NoGradGuard no_grad;
        torch::AutoGradMode enable_grad(false);
  // get inputs
  std::vector<torch::Tensor> states;
  std::vector<std::promise<return_type>> promises;

  bool timeout = false;
  while (states.size() < this->batch_size && !timeout) {
    // pop task
    {
      std::unique_lock<std::mutex> lock(this->lock);
      if (this->cv.wait_for(lock, 1ms,
                            [this] { return this->tasks.size() > 0; })) {
        auto task = std::move(this->tasks.front());
        states.emplace_back(std::move(task.first));
        promises.emplace_back(std::move(task.second));

        this->tasks.pop();

      } else {
        // timeout
        // std::cout << "timeout" << std::endl;
        timeout = true;
      }
    }
  }

  // inputs empty
  if (states.size() == 0) {
    return;
  }

  Tensor inputs = USE_GPU ? cat(states, 0).to(at::kCUDA) : cat(states, 0);
  

   auto result = this->module->forward(inputs);
   //std::cout << y.requires_grad() << std::endl; // prints `false`

  
  Tensor p_batch = result.first.exp().toType(kFloat32).to(at::kCPU);
  Tensor v_batch = result.second.toType(kFloat32).to(at::kCPU);

  // set promise value
  for (unsigned int i = 0; i < promises.size(); i++) {
    torch::Tensor p = p_batch[i];
    torch::Tensor v = v_batch[i];

    std::vector<double> prob(static_cast<float*>(p.data_ptr()),
                             static_cast<float*>(p.data_ptr()) + p.size(0));
    std::vector<double> value{v.item<float>()};

    return_type temp{std::move(prob), std::move(value)};

    promises[i].set_value(std::move(temp));
  }
  }
}

void NeuralNetwork::train(board_buff_type board_buffer, p_buff_type p_buffer, v_buff_type v_buffer){
  
    int size = board_buffer.size();
    unsigned seed = rand();
        //std::chrono::system_clock::now().time_since_epoch().count();
    auto e = std::default_random_engine(seed);
    shuffle(board_buffer.begin(), board_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(p_buffer.begin(), p_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(v_buffer.begin(), v_buffer.end(), e);

    //std::random_shuffle(board_buffer.begin(), board_buffer.end());
    //board_buff_type rand_board_buffer = CustomType::board_buff_type(size);
    //std::vector<int> rand_order(size);
    //for (int i = 0; i < size; i++) {
    //    rand_order.push_back (i);
    //}
    //std::random_shuffle(rand_order.begin(), rand_order.end());
    //std::sort(board_buffer.begin(), board_buffer.end(), rand_order);
    //unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count();
    //std::shuffle (rand_order.begin (), rand_order.end (), std::default_random_engine(seed));
  //for (int i = 0; i < size; i++) {
  //    rand_board_buffer.push_back(board_buffer(rand_order[i]));
  //}
  //board_buffer.clear();
  int batch_size = 7;
  for (int pt = 0; pt < size - batch_size; pt += batch_size) {
      std::cout << "train pt = " << pt << std::endl;
      Tensor inputs = cat(board_buff_type(board_buffer.begin() + pt, board_buffer.begin() + pt + batch_size), 0);
      //std::cout << "inputs dim = " << inputs.dim() << std::endl;
      //std::cout << "inputs size = " << inputs.size(0) << " " << inputs.size(1) << std::endl;

      Tensor vs = torch::tensor(v_buff_type(v_buffer.begin() + pt, v_buffer.begin() + pt + batch_size)).unsqueeze(1);

      std::vector<Tensor> p_tensor(batch_size);
      for (int i = 0; i < batch_size; i++) {
          //v_tensor[i] = torch::tensor(v_buffer[pt + i]);
          p_tensor[i] = torch::tensor(p_buffer[pt + i]).unsqueeze(0);
      }
      //std::cout << "vector size = " << p_tensor.size() << std::endl;
      Tensor ps = cat(p_tensor, 0);
      //Tensor vs = cat(v_tensor, 0);
      optimizer->zero_grad();
      auto result = this->module->forward(inputs);
      
      //Tensor log_ps, Tensor vs, const Tensor target_ps, const Tensor target_vs
      Tensor loss = alpha_loss(result.first, result.second, ps, vs);
      loss.backward();
      optimizer->step();
  }
}

Tensor func(int n)
{
    return tensor(n);
}
