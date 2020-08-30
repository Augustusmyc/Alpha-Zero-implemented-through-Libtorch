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
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
  MCTS m(nn, 4, 3, 16, 0.3, g->get_action_size());
  std::pair<int,int> game_state;
  game_state = g->get_game_status();
  std::cout << "begin !!" << std::endl;
  int step = 0;
  board_buff_type *cur_board_buffer = new board_buff_type();
  v_buff_type *cur_color_buff_type = new v_buff_type();
  p_buff_type *cur_p_buffer = new p_buff_type();

  while (game_state.first == 0) {
    //int res = m.get_best_action(g.get());
    double temp = step < 10 ? 1 : 0;
    auto action_probs = m.get_action_probs(g.get(), temp);
    cur_p_buffer->emplace_back(action_probs);
    cur_board_buffer->emplace_back(NeuralNetwork::transorm_gomoku_to_Tensor(g.get()));
    cur_color_buff_type->emplace_back(g->get_current_color());
    int res = m.get_action_by_sample(action_probs);
    m.update_with_move(res);
    g->execute_move(res);
    game_state = g->get_game_status();
    step++;
  }
  if (game_state.second != 0) {
      {
          std::lock_guard<std::mutex> lock(this->lock);
          for (int i = 0; i < cur_p_buffer->size(); i++) {
              this->board_buffer->emplace_back(cur_board_buffer->at(i));
              this->v_buffer->emplace_back(cur_color_buff_type->at(i) * game_state.second);
              this->p_buffer->emplace_back(cur_p_buffer->at(i));
          }
      }
  }

  std::cout << "total step num = " << step << std::endl;
  //return { *board_buffer , *p_buffer , *cur_color_buff_type };
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
