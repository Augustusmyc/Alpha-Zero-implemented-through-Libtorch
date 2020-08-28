#include <play.h>
#include <mcts.h>
#include <gomoku.h>
#include <common.h>
#include <libtorch.h>

using namespace customType;

SelfPlay::SelfPlay(NeuralNetwork *nn):
        p_buffer(new p_buff_type()),
        board_buffer(new board_buff_type()),
        v_buffer(new v_buff_type()),
        nn(nn),
        thread_pool(new ThreadPool(3))
        {}

void SelfPlay::play(){
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, 5, 1);
  MCTS m(nn, 4, 3, 16, 0.3, g->get_action_size());
  std::pair<int,int> game_state;
  game_state = g->get_game_status();
  std::cout << "begin !!" << std::endl;
  int step = 0;
  while (game_state.first == 0) {
    //int res = m.get_best_action(g.get());
    double temp = step < 10 ? 1 : 0;
    auto action_probs = m.get_action_probs(g.get(), temp);
    {
       std::lock_guard<std::mutex> lock(this->lock);
       p_buffer->emplace_back(action_probs);   //TODO v board
       board_buffer->emplace_back(NeuralNetwork::transorm_gomoku_to_Tensor(g.get()));
    }
    int res = m.get_action_by_sample(action_probs);
    m.update_with_move(res);
    g->execute_move(res);
    game_state = g->get_game_status();
    step++;
  }
  for (int i = 0; i < p_buffer->size(); i++) {
        v_buffer->emplace_back(game_state.second);
    }

  std::cout << "total step num = " << step << std::endl;
}

std::tuple<board_buff_type, p_buff_type, v_buff_type> SelfPlay::self_play_for_train(unsigned int game_num){
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(&SelfPlay::play, this));
        futures.emplace_back(std::move(future));
    }
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
    }
    return { *this->board_buffer , *this->p_buffer ,*this->v_buffer };
}
