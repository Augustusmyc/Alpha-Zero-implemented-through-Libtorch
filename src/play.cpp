#include <play.h>
#include <mcts.h>
#include <gomoku.h>
#include <common.h>
#include <libtorch.h>

#ifdef USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace customType;

SelfPlay::SelfPlay(NeuralNetwork *nn):
        p_buffer(new p_buff_type()),
        board_buffer(new board_buff_type()),
        v_buffer(new v_buff_type()),
        nn(nn),
        thread_pool(new ThreadPool(NUM_TRAIN_THREADS))
        {}

std::pair<int, int> SelfPlay::self_play_for_eval(NeuralNetwork *a, NeuralNetwork *b) {
    int a_win_count = 0;
    int b_win_count = 0;
    //int tie = 0;
    MCTS ma(a, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    MCTS mb(b, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    for (int episode = 0; episode < 10; episode++) {
        int step = 0;
        auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
        std::pair<int, int> game_state = g->get_game_status();
        //std::cout << episode << " th game!!" << std::endl;
        while (game_state.first == 0) {
            bool cur_net = (step + episode) % 2 == 0;
            int res = cur_net ? ma.get_best_action(g.get()) : mb.get_best_action(g.get());
            ma.update_with_move(res);
            mb.update_with_move(res);
            g->execute_move(res);
            game_state = g->get_game_status();
            step++;
        }
        std::cout << "eval: total step num = " << step << std::endl;
        if ((game_state.second == BLACK && episode % 2 == 0) || (game_state.second == WHITE && episode % 2 == 1)) a_win_count++;
        else if ((game_state.second == BLACK && episode % 2 == 1) || (game_state.second == WHITE && episode % 2 == 0)) b_win_count++;
        //else if (game_state.second == 0) tie++;
#ifdef USE_GPU
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif
        
    }
    return { a_win_count ,b_win_count };
}


void SelfPlay::play(){
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
  MCTS m(nn, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
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
  //if (game_state.second != 0) {
      {
          std::lock_guard<std::mutex> lock(this->lock);
          for (int i = 0; i < cur_p_buffer->size(); i++) {
              this->board_buffer->emplace_back(cur_board_buffer->at(i));
              this->v_buffer->emplace_back(cur_color_buff_type->at(i) * game_state.second);
              this->p_buffer->emplace_back(cur_p_buffer->at(i));
          }
      }
  //}

  std::cout << "total step num = " << step << " winner = " << game_state.second <<std::endl;
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
