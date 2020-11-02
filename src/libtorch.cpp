#include <libtorch.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <ATen/cuda/CUDAGuard.h>

#include <torch/torch.h>
#include <iostream>
#include <common.h>
#include <random>

#ifdef USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#ifdef JIT_MODE
#define TS std::vector<torch::jit::IValue>
#else
#define TS Tensor
#endif // JIT_MODE


using namespace customType;
using namespace std::chrono_literals;
using namespace torch;

nn::Conv2d conv3x3(int64_t in_channels, int64_t out_channels, unsigned int stride=1){
  return nn::Conv2d(nn::Conv2dOptions(/*in_channel=*/in_channels, out_channels, /*kernel_size=*/3).stride(stride).padding(1).bias(false));
}

struct ResidualBlock:nn::Module {
  ResidualBlock(){
    //only for initializing array
  }

  ResidualBlock(int64_t in_channels, int64_t out_channels, unsigned int stride,int global_count){
    conv1 = register_module("conv1"+str(global_count),conv3x3(in_channels, out_channels, stride));
    bn1 = register_module("bn1" + str(global_count), nn::BatchNorm2d(out_channels));
    conv2 = register_module("conv2" + str(global_count),conv3x3(out_channels, out_channels));
    bn2 = register_module("bn2" + str(global_count), nn::BatchNorm2d(out_channels));
    // relu = register_module("relu", nn::ReLU(true));
    this->out_channels = out_channels;
    downsample = false;
    if (in_channels != out_channels || stride != 1){
      downsample = true;
      downsample_conv = register_module("downsample_conv" + str(global_count),conv3x3(in_channels, out_channels, stride=stride));
      downsample_bn = register_module("downsample_bn" + str(global_count), nn::BatchNorm2d(out_channels));
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
    res_list[0] = ResidualBlock(3, num_channels,1,-999);
    for (unsigned int i = 1; i<num_layers; i++){
        res_list[i] = ResidualBlock(num_channels, num_channels,1,i);
     }
#ifdef USE_GPU
        for (unsigned int i = 0; i < num_layers; i++) {
            res_list[i].to(torch::kCUDA);
        }
#endif   
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


NeuralNetwork::NeuralNetwork(std::string model_path, unsigned int batch_size)
    : //module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
      batch_size(batch_size),
      running(true),
      //optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4)),
      loop(nullptr) {
#ifdef JIT_MODE
    module = std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()));
  //this->optimizer = new torch::optim::Adam(this->module->parameters(), torch::optim::AdamOptions(LR));
#else
    module = std::make_shared<AlphaZeroNet>(
                AlphaZeroNet(/*num_layers=*/NUM_LAYERS,/*num_channels=*/NUM_CHANNELS,/*n=*/BORAD_SIZE,/*action_size=*/BORAD_SIZE * BORAD_SIZE));
    module->load_weights(model_path.c_str());
#endif
#ifdef USE_GPU
    // move to CUDA
    this->module->to(at::kCUDA);
#endif

   //this->optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
   //auto optimizer = torch::optim::Adam(this->module->parameters(), 0.01);

  // run infer thread
  this->loop = std::make_unique<std::thread>([this] {
    while (this->running) {
      this->infer();
    }
  });
}

//NeuralNetwork::NeuralNetwork(unsigned int batch_size)
//    : //module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
//    batch_size(batch_size),
//    running(true),
//    //optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4)),
//    loop(nullptr) {
//    module = std::make_shared<AlphaZeroNet>(
//        AlphaZeroNet(/*num_layers=*/NUM_LAYERS,/*num_channels=*/NUM_CHANNELS,/*n=*/BORAD_SIZE,/*action_size=*/BORAD_SIZE * BORAD_SIZE));
//
//    //this->optimizer = new torch::optim::Adam(this->module->parameters(), torch::optim::AdamOptions(LR));
//
//#ifdef USE_GPU
//    // move to CUDA
//    this->module->to(at::kCUDA);
//#endif
//
//    //this->optimizer(this->module->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
//    //auto optimizer = torch::optim::Adam(this->module->parameters(), 0.01);
//
//   // run infer thread
//    this->loop = std::make_unique<std::thread>([this] {
//        while (this->running) {
//            this->infer();
//        }
//        });
//}

#ifndef JIT_MODE
void NeuralNetwork::save_weights(string model_path) {
    torch::save(this->module, model_path.c_str());
}

void NeuralNetwork::load_weights(string model_path) {
    torch::load(this->module, model_path.c_str());
}
#endif





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

Tensor NeuralNetwork::transorm_board_to_Tensor(board_type board, int last_move, int cur_player) {
    std::vector<int> board0;
    for (unsigned int i = 0; i < BORAD_SIZE; i++) {
        board0.insert(board0.end(), board[i].begin(), board[i].end());
    }

    torch::Tensor temp =
        torch::from_blob(&board0[0], { 1, 1, BORAD_SIZE, BORAD_SIZE }, torch::dtype(torch::kInt32));

    torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
    torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

    if (cur_player == -1) {
        std::swap(state0, state1);
    }

    torch::Tensor state2 =
        torch::zeros({ 1, 1, BORAD_SIZE, BORAD_SIZE }, torch::dtype(torch::kFloat32));

    if (last_move != -1) {
        state2[0][0][last_move / BORAD_SIZE][last_move % BORAD_SIZE] = 1;
    }

    //torch::Tensor states = torch::cat({ state0, state1 }, 1);
    return cat({ state0, state1, state2 }, 1);
}

Tensor NeuralNetwork::transorm_gomoku_to_Tensor(Gomoku* gomoku){
  return NeuralNetwork::transorm_board_to_Tensor(gomoku->get_board(), gomoku->get_last_move(), gomoku->get_current_color());
}

void NeuralNetwork::infer() {
    {
        this->module->eval();
        torch::NoGradGuard no_grad;
        //torch::AutoGradMode enable_grad(false);
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
        //std::cout << "timeout" << std::endl;
        timeout = true;
      }
    }
  }

  // inputs empty
  if (states.size() == 0) {
    return;
  }
#ifdef USE_GPU
    TS inputs{ cat(states, 0).to(at::kCUDA) };
#else
    TS inputs = cat(states, 0);
#endif

#ifdef JIT_MODE
    auto result = this->module->forward(inputs).toTuple();
    torch::Tensor p_batch = result->elements()[0]
        .toTensor()
        .exp()
        .toType(torch::kFloat32)
        .to(at::kCPU);
    torch::Tensor v_batch =
        result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);
#else
    auto result = this->module->forward(inputs);
    //std::cout << y.requires_grad() << std::endl; // prints `false`


    Tensor p_batch = result.first.exp().toType(kFloat32).to(at::kCPU);
    Tensor v_batch = result.second.toType(kFloat32).to(at::kCPU);
#endif



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
