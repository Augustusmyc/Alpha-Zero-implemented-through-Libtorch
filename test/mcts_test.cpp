#include <iostream>
#include <mcts.h>

#include <torch/script.h>
#include <libtorch.h>

int main() {
  const int board_size = 15;
  auto g = std::make_shared<Gomoku>(board_size, 5, 1);
  //Gomoku g(15, 5, 1);
  //g.execute_move(12);
  //g->execute_move(12);
  //g->execute_move(13);
  //g->execute_move(14);
  //g->execute_move(15);
  //g->execute_move(16);
  // g->execute_move(17);
  // g->execute_move(18);
  // g->execute_move(19);
  // g->render();
  
  // Deserialize the ScriptModule from a file using torch::jit::load().
  //std::shared_ptr<torch::jit::script::Module> module = torch::jit::load("../test/models/checkpoint.pt");
  //torch::jit::script::Module module = torch::jit::load("../test/models/checkpoint.pt");
  
  //NeuralNetwork *module = new NeuralNetwork("/dataspace/azgomu/models/best_checkpoint.pt", true, 1600);
  NeuralNetwork *module = new NeuralNetwork(true, 1600);
  module->save_weights("net.pt");
  module->load_weights("net.pt");
  MCTS m(module, 4, 3, 1600, 0.3, g->get_action_size());

  std::cout << "RUNNING" << std::endl;

  
  char move_ic;
  int move_j;
  std::vector<int> game_state;
  bool is_illlegal = true;

  while (true) {
    int res = m.get_best_action(g.get());
    m.update_with_move(res);
    g->execute_move(res);
    g->render();
    game_state = g->get_game_status();
    if (game_state[0] != 0) break;
    int x, y;
    printf("your move: \n");
    std::cin >> move_ic >> move_j;
    x = move_ic - 'A';
    y = move_j - 1;
    is_illlegal = g->is_illegal(x,y);
    while (is_illlegal){
      printf("Illegal move ! Please input \"character\" and \"number\" such as A 1 and ensure the position is empty !\n");
      printf("move again: \n");
      std::cin >> move_ic >> move_j;
      x = move_ic - 'A';
      y = move_j - 1;
      is_illlegal = g->is_illegal(x,y);
    }
    int my_move = x*board_size+y;

    m.update_with_move(my_move);
    g->execute_move(my_move);
    g->render();
    std::vector<int> game_state = g->get_game_status();
    if (game_state[0] != 0) break;


    // std::for_each(res.begin(), res.end(),
    //               [](double x) { std::cout << x << ","; });
    // std::cout << std::endl;
    // m.update_with_move(-1);
  }
  std::cout << "winner num = " << game_state[1] << std::endl;
  return 0;
}

