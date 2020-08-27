#include <play.h>
#include <mcts.h>
#include <gomoku.h>
#include <common.h>
#include <libtorch.h>

SelfPlay::SelfPlay(NeuralNetwork *nn):
        p_buffer(new CustomType::p_buff_type()),
        board_buffer(new CustomType::board_buff_type()),
        nn(nn),
        thread_pool(new ThreadPool(3))
        {}

void SelfPlay::play(){
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, 5, 1);
  MCTS m(nn, 4, 3, 160, 0.3, g->get_action_size());
  std::vector<int> game_state;
  game_state = g->get_game_status();
  std::cout << "begin !!" << std::endl;
  int step = 0;
  while (game_state[0] == 0) {
    //int res = m.get_best_action(g.get());
    double temp = step < 10 ? 1 : 0;
    auto action_probs = m.get_action_probs(g.get(), temp);
    {
       std::lock_guard<std::mutex> lock(this->lock);
       p_buffer->emplace_back(action_probs);   //TODO v board
       //board_buffer->emplace_back(transorm_gomoku_to_Tensor(g.get()));//->get_board());
    }
    int res = m.get_action_by_sample(action_probs);
    m.update_with_move(res);
    g->execute_move(res);
    game_state = g->get_game_status();
    step++;
  }
  std::cout << "total step num = " << step << std::endl;
}

CustomType::p_buff_type SelfPlay::self_play_for_train(unsigned int game_num){
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(&SelfPlay::play, this));
        futures.emplace_back(std::move(future));
    }
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
    }
    return *this->p_buffer;
}

CustomType::board_buff_type SelfPlay::get_buffer(){
   return *this->board_buffer;
}
