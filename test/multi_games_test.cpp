#include <iostream>
#include <mcts.h>

#include <torch/script.h>
#include<torch/torch.h>
#include <libtorch.h>
#include <play.h>

int main() {
  NeuralNetwork *model = new NeuralNetwork("/dataspace/azgomu/models/best_checkpoint.pt", true, 1024);
  //torch::optim::SGD optimizer(model->module->parameters(), /*lr=*/0.01);
  SelfPlay *sp = new SelfPlay(model);
  sp->self_play_for_train(3);
  auto p_buffer = sp->get_buffer();
  return 0;
}

