#include <iostream>
#include <mcts.h>

#include <torch/script.h>
#include<torch/torch.h>
#include <libtorch.h>
#include <play.h>

int main() {
  NeuralNetwork *model = new NeuralNetwork(8);
  //torch::optim::SGD optimizer(model->module->parameters(), /*lr=*/0.01);
  SelfPlay *sp = new SelfPlay(model);
  auto train_buffer = sp->self_play_for_train(3);
  //std::cout << train_buffer[0] << std::endl;
  return 0;
}

