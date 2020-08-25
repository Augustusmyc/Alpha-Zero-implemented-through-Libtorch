#include <torch/torch.h>
#define BOARD_SIZE 7
#define HAS_CUDA false

// This file is independent of other cpp and h files.

// Define a new Module.
using namespace torch;

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
    myrelu = register_module("myrelu", nn::ReLU(true));
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
    //x = torch::relu(fc1->forward(x.reshape({x.size(0), 120})));
    //x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    Tensor out = myrelu(bn1(conv1(x)));
    out = bn2(conv2(out));
    if (downsample){
      residual = downsample_conv(residual);
      residual = downsample_bn((residual));
    }

    out += residual;
    out = myrelu(out);
    return out;
  }

  // Use one of many "standard library" modules.
  nn::Conv2d conv1{nullptr};
  nn::Conv2d conv2{nullptr};
  nn::BatchNorm2d bn1{nullptr};
  nn::BatchNorm2d bn2{nullptr};
  nn::BatchNorm2d downsample_bn{nullptr};
  nn::Conv2d downsample_conv{nullptr};
  nn::ReLU myrelu{nullptr};
  bool downsample;
  int64_t out_channels;
};



struct AlphaZeroNet : nn::Module {
  AlphaZeroNet(const unsigned int num_layers, int64_t num_channels, unsigned int n, unsigned int action_size){
    // residual block
    ResidualBlock *res_list = new ResidualBlock[num_layers]();//  ResidualBlock(3, num_channels);
    res_list[0] = ResidualBlock(3, num_channels);
    
    for (unsigned int i = 1; i<num_layers; i++){
        res_list[i] = ResidualBlock(num_channels, num_channels);
     }
    if (HAS_CUDA) {
        for (unsigned int i = 0; i < num_layers; i++) {
            res_list[i].to(torch::kCUDA);
        }
    }

    // ResidualBlock res_list[] = {ResidualBlock(3, num_channels),ResidualBlock(num_channels, num_channels)};
    
    // res_list[0].to(torch::kCUDA);
    // res_list[1].to(torch::kCUDA); 
    res_layers = nn::Sequential(*res_list);

    //  policy head
    p_conv = register_module("p_conv", nn::Conv2d(
      nn::Conv2dOptions(/*in_channel=*/num_channels, 4, /*kernel_size=*/1).padding(0).bias(false)));
    p_bn = register_module("p_bn",nn::BatchNorm2d(4));
    myrelu = register_module("myrelu", nn::ReLU(true));
    p_fc = register_module("p_fc",nn::Linear(4 * n * n, action_size));
    p_log_softmax = register_module("p_log_softmax", nn::LogSoftmax(/*dim=*/1));
    

     // value head
    v_conv = register_module("v_conv",nn::Conv2d(nn::Conv2dOptions(/*in_channel=*/num_channels, 2, /*kernel_size=*/1).padding(0).bias(false)));
    v_bn = register_module("v_bn",nn::BatchNorm2d(2));

    v_fc1 = register_module("v_fc1", nn::Linear(2 * n * n, 256));
    v_fc2 = register_module("v_fc2", nn::Linear(256, 1));
  }

  // Implement the Net's algorithm.
 std::pair<Tensor,Tensor> forward(Tensor inputs) {
    // Use one of many tensor manipulation functions.
    //x = torch::relu(fc1->forward(x.reshape({x.size(0), 120})));
    //x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());

    Tensor out = res_layers->forward(inputs);

    // policy head
    Tensor p = p_conv(out);
    p = p_bn(p);
    p = myrelu(p);

    p = p_fc(p.view({p.size(0), -1}));
    p = p_log_softmax(p);

    // value head
    Tensor v = v_conv(out);
    v = v_bn(v);
    v = myrelu(v);

    v = v_fc1(v.view({v.size(0), -1}));
    v = myrelu(v);
    v = tanh(v_fc2(v));

    //x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return std::make_pair(p,v);
  }

  // Use one of many "standard library" modules.
  nn::Sequential res_layers{nullptr};
  nn::Conv2d p_conv{nullptr};
  nn::Conv2d v_conv{nullptr};
  nn::BatchNorm2d p_bn{nullptr};
  nn::BatchNorm2d v_bn{nullptr};
  nn::Linear p_fc{nullptr};
  nn::Linear v_fc1{nullptr};
  nn::Linear v_fc2{nullptr};
  nn::ReLU myrelu{nullptr};
  nn::LogSoftmax p_log_softmax{nullptr};
  // ResidualBlock *res_list;
};

int main() {

  //Tensor t1 = torch::arange(18).reshape({2,3,3});
  //// Tensor t2 = torch::flip(t1,{0，1});
  //Tensor t2 = torch::rot90(t1,/*k=*/1, /*dims=*/{1,2});
  //Tensor t3 = torch::rot90(t1,/*k=*/3, /*dims=*/{1,2});
  //Tensor t = torch::cat({t1, t2, t3}, 0);
  //std::cout << "t"<<t.slice(/*dim=*/1, /*start=*/0, /*end=*/15) << '\n';

   auto net = std::make_shared<AlphaZeroNet>(
     AlphaZeroNet(/*num_layers=*/4,/*num_channels=*/256,/*n=*/BOARD_SIZE,/*action_size=*/BOARD_SIZE*BOARD_SIZE));
   


   optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
   Tensor input = torch::randn({ 6, 3, BOARD_SIZE, BOARD_SIZE }).toType(torch::kFloat32);


   Tensor target_v = torch::ones({ 6 }).toType(torch::kFloat32);
   Tensor target_p = torch::zeros({ 6, BOARD_SIZE*BOARD_SIZE }).toType(kFloat32);
   if (HAS_CUDA) {
       net->to(torch::kCUDA);
       input.to(torch::kCUDA);
       target_v.to(torch::kCUDA);
       target_p.to(torch::kCUDA);
   }


   int c = 0;
   while(c<5){
     auto result = net->forward(input);
     Tensor log_p = result.first;
     Tensor v = result.second;
    
     Tensor loss = alpha_loss(log_p, v, target_p,target_v);
     loss.backward();
     optimizer.step();
     std::cout <<" Loss: " << loss.item<float>() << std::endl;
     c++;
   }

   net->eval();
   auto result = net->forward(input);
   Tensor p = result.first;
   Tensor v = result.second;

   std::cout << "p"<<p.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
   std::cout << "v"<<v.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  
  

  
   torch::save(net,"net.pt");

   auto new_net = std::make_shared<AlphaZeroNet>(
       AlphaZeroNet(/*num_layers=*/4,/*num_channels=*/256,/*n=*/BOARD_SIZE,/*action_size=*/BOARD_SIZE*BOARD_SIZE));
   new_net->eval();
   if (HAS_CUDA) {
       new_net->to(torch::kCUDA);
   }
   torch::load(new_net,"net.pt");
   Tensor new_pred = new_net->forward(input).first;
   std::cout << "new_pred"<<new_pred.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
