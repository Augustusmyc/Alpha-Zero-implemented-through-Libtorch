#include <iostream>
#include <mcts.h>

#include <torch/script.h>
#include <libtorch.h>
#include <common.h>

using namespace std;

int main(int argc, char* argv[]) {
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
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
  
  NeuralNetwork* module = nullptr;
  bool ai_black = true;
  if (argc <= 1) {
      // cout << "Do not load weights. AI color = BLACK." << endl;
      // module = new NeuralNetwork(BATCH_SIZE);
      cout << "Warning: Find No weight path and color, assume they are ./weights/0.pt and 1 (AI color:Black)" << endl;
      module = new NeuralNetwork("./weights/0.pt", BATCH_SIZE);
  }
  else {
      ai_black = strcmp(argv[2], "1") == 0 ? true : false;
      string color = ai_black ? "BLACK" : "WHITE";
      cout << "Load weights: "<< argv[1] << "  AI color: " << color << endl;
      module = new NeuralNetwork(argv[1],BATCH_SIZE);
  }
  //module->save_weights("net.pt");
  
  MCTS m(module, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);

  std::cout << "Running..." << std::endl;

  
  char move_ic;
  int move_j;
  bool is_illlegal = true;
  std::pair<int, int> game_state;
  if (ai_black) {
      int res = m.get_best_action(g.get());
      m.update_with_move(res);
      g->execute_move(res);
  }

  while (true) {
    g->render();
    game_state = g->get_game_status();
    if (game_state.first != 0) break;
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
    int my_move = x * BORAD_SIZE + y;
    m.update_with_move(my_move);
    g->execute_move(my_move);
    game_state = g->get_game_status();
    if (game_state.first != 0) {
        g->render();
        break;
    }

    int res = m.get_best_action(g.get());
    m.update_with_move(res);
    g->execute_move(res);


    // std::for_each(res.begin(), res.end(),
    //               [](double x) { std::cout << x << ","; });
    // std::cout << std::endl;
    // m.update_with_move(-1);
  }
  std::cout << "winner num = " << game_state.second << std::endl;
  return 0;
}

