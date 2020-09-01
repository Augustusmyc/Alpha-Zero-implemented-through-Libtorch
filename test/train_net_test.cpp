#include <iostream>

#include <mcts.h>

#include <torch/script.h>
#include<torch/torch.h>
#include <libtorch.h>
#include <play.h>
#include <common.h>

int main() {

    // system("mkdir weights");
    #ifdef _WIN32
    system("mkdir .\\weights");
    #elif __linux__
    system("mkdir ./weights");
    #endif

    NeuralNetwork* model = new NeuralNetwork(BATCH_SIZE);
    NeuralNetwork* old_best_model = new NeuralNetwork(BATCH_SIZE);
    model->save_weights("./weights/0.pt");
    int best_weight = 0;
    //torch::optim::SGD optimizer(model->module->parameters(), /*lr=*/0.01);
    int current_weight = 1;
    std::ostringstream old_best_path;
    std::ostringstream new_path;
    while (true) {
        SelfPlay* sp = new SelfPlay(model);
        auto train_buffer = sp->self_play_for_train(NUM_TRAIN_THREADS);
        //std::cout << "3 train size = " << std::get<0>(train_buffer).size() << " " <<
        //    std::get<1>(train_buffer).size() << " " << std::get<2>(train_buffer).size() << std::endl;
        model->train(std::get<0>(train_buffer), std::get<1>(train_buffer), std::get<2>(train_buffer));

        new_path.str("");
        new_path << "./weights/" << current_weight << ".pt";
        model->save_weights(new_path.str());

        old_best_path.str("");
        old_best_path << "./weights/" << best_weight << ".pt";
        old_best_model->load_weights(old_best_path.str());

        auto win_table = sp->self_play_for_eval(old_best_model, model);
        std::cout << "old win:" << win_table.first << " & new win:" << win_table.second << std::endl;
        if (win_table.second > win_table.first + 2) {
            std::cout << "New best model generated!!" << " Current weight = " << current_weight << std::endl;
            int best_weight = current_weight;
        }
        current_weight++;
    }
  //Tensor t1 = torch::arange(18).reshape({2,3,3});
  //// Tensor t2 = torch::flip(t1,{0，1});
  //Tensor t2 = torch::rot90(t1,/*k=*/1, /*dims=*/{1,2});
  //Tensor t3 = torch::rot90(t1,/*k=*/3, /*dims=*/{1,2});
  //Tensor t = torch::cat({t1, t2, t3}, 0);
  //std::cout << "t"<<t.slice(/*dim=*/1, /*start=*/0, /*end=*/15) << '\n';

   //auto net = std::make_shared<AlphaZeroNet>(
   //  AlphaZeroNet(/*num_layers=*/4,/*num_channels=*/256,/*n=*/BOARD_SIZE,/*action_size=*/BOARD_SIZE*BOARD_SIZE));
   //


   //optim::SGD optimizer(net->parameters(), /*lr=*/0.01);
   //Tensor input = torch::randn({ 6, 3, BOARD_SIZE, BOARD_SIZE }).toType(torch::kFloat32);


   //Tensor target_v = torch::ones({ 6 }).toType(torch::kFloat32);
   //Tensor target_p = torch::zeros({ 6, BOARD_SIZE*BOARD_SIZE }).toType(kFloat32);
   //if (HAS_CUDA) {
   //    net->to(torch::kCUDA);
   //    input.to(torch::kCUDA);
   //    target_v.to(torch::kCUDA);
   //    target_p.to(torch::kCUDA);
   //}


   //int c = 0;
   //while(c<5){
   //  auto result = net->forward(input);
   //  Tensor log_p = result.first;
   //  Tensor v = result.second;
   // 
   //  Tensor loss = alpha_loss(log_p, v, target_p,target_v);
   //  loss.backward();
   //  optimizer.step();
   //  std::cout <<" Loss: " << loss.item<float>() << std::endl;
   //  c++;
   //}

   //net->eval();
   //auto result = net->forward(input);
   //Tensor p = result.first;
   //Tensor v = result.second;

   //std::cout << "p"<<p.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
   //std::cout << "v"<<v.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
  
  

  
   //torch::save(net,"net.pt");

   //auto new_net = std::make_shared<AlphaZeroNet>(
   //    AlphaZeroNet(/*num_layers=*/4,/*num_channels=*/256,/*n=*/BOARD_SIZE,/*action_size=*/BOARD_SIZE*BOARD_SIZE));
   //new_net->eval();
   //if (HAS_CUDA) {
   //    new_net->to(torch::kCUDA);
   //}
   //torch::load(new_net,"net.pt");
   //Tensor new_pred = new_net->forward(input).first;
   //std::cout << "new_pred"<<new_pred.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
