#include <play.h>
#include <mcts.h>
#include <gomoku.h>
#include <common.h>
#include <libtorch.h>

#include<iostream>
#include<fstream>

#ifdef USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace customType;
using namespace std;

SelfPlay::SelfPlay(NeuralNetwork *nn):
        //p_buffer(new p_buff_type()),
        //board_buffer(new board_buff_type()),
        //v_buffer(new v_buff_type()),
        nn(nn),
        thread_pool(new ThreadPool(NUM_TRAIN_THREADS))
        {}

//std::pair<int, int> SelfPlay::self_play_for_eval(NeuralNetwork *a, NeuralNetwork *b) {
//    int a_win_count = 0;
//    int b_win_count = 0;
//    //int tie = 0;
//    MCTS ma(a, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
//    MCTS mb(b, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
//    for (int episode = 0; episode < 4; episode++) {
//        int step = 0;
//        auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
//        std::pair<int, int> game_state = g->get_game_status();
//        //std::cout << episode << " th game!!" << std::endl;
//        while (game_state.first == 0) {
//            bool cur_net = (step + episode) % 2 == 0;
//            int res = cur_net ? ma.get_best_action(g.get()) : mb.get_best_action(g.get());
//            ma.update_with_move(res);
//            mb.update_with_move(res);
//            g->execute_move(res);
//            game_state = g->get_game_status();
//            step++;
//        }
//        cout << "eval: total step num = " << step << endl;
//        if ((game_state.second == BLACK && episode % 2 == 0) || (game_state.second == WHITE && episode % 2 == 1)) a_win_count++;
//        else if ((game_state.second == BLACK && episode % 2 == 1) || (game_state.second == WHITE && episode % 2 == 0)) b_win_count++;
//        //else if (game_state.second == 0) tie++;
//#ifdef USE_GPU
//        c10::cuda::CUDACachingAllocator::emptyCache();
//#endif
//        
//    }
//    return { a_win_count ,b_win_count };
//}


void SelfPlay::play(unsigned int id){
  auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
  MCTS *m = new MCTS(nn, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
  std::pair<int,int> game_state;
  game_state = g->get_game_status();
  //std::cout << "begin !!" << std::endl;
  int step = 0;
  board_buff_type board_buffer(BUFFER_LEN, vector<vector<int>>(BORAD_SIZE, vector<int>(BORAD_SIZE)));
  v_buff_type v_buffer(BUFFER_LEN);
  p_buff_type p_buffer(BUFFER_LEN, vector<float>(BORAD_SIZE * BORAD_SIZE));// = new p_buff_type();
  vector<int> col_buffer(BUFFER_LEN);
  vector<int> last_move_buffer(BUFFER_LEN);


  while (game_state.first == 0) {
      //int res = m.get_best_action(g.get());
        double temp = step < 10 ? 1 : 0;
        auto action_probs = m->get_action_probs(g.get(), temp);
        board_type board = g->get_board();
        for (int i = 0; i < BORAD_SIZE * BORAD_SIZE; i++) {
            p_buffer[step][i] = action_probs[i];
        }
        for (int i = 0; i < BORAD_SIZE; i++) {
            for (int j = 0; j < BORAD_SIZE; j++) {
                board_buffer[step][i][j] = board[i][j];
            }
        }
        col_buffer[step] = g->get_current_color();
        last_move_buffer[step] = g->get_last_move();

        // diri noise
        static std::gamma_distribution<float> gamma(0.3f, 1.0f);
        static std::default_random_engine rng(std::time(nullptr));
        std::vector<int> lm = g->get_legal_moves();
        float sum = 0;
        for (unsigned int i = 0; i < lm.size(); i++) {
            if (lm[i]) {
                action_probs[i] += DIRI * gamma(rng);
                sum += action_probs[i];
            }
        }
        for (unsigned int i = 0; i < lm.size(); i++) {
            if (lm[i]) {
                action_probs[i] /= sum;
            }
        }


        int res = m->get_action_by_sample(action_probs);
        m->update_with_move(res);
        g->execute_move(res);
        game_state = g->get_game_status();
        step++;
    }
      cout << "total step num = " << step << " winner = " << game_state.second << endl;

      ofstream bestand;
      bestand.open("./data/data_"+str(id), ios::out | ios::binary);
      bestand.write(reinterpret_cast<char*>(&step), sizeof(int));

      for (int i = 0; i < step; i++) {
          for (int j = 0; j < BORAD_SIZE; j++) {
              bestand.write(reinterpret_cast<char*>(&board_buffer[i][j][0]), BORAD_SIZE * sizeof(int));
          }
      }

      for (int i = 0; i < step; i++) {
          bestand.write(reinterpret_cast<char*>(&p_buffer[i][0]), BORAD_SIZE * BORAD_SIZE * sizeof(float));
          v_buffer[i] = col_buffer[step] * game_state.second;
      }

      bestand.write(reinterpret_cast<char*>(&v_buffer[0]), step * sizeof(int));
      bestand.write(reinterpret_cast<char*>(&col_buffer[0]), step * sizeof(int));
      bestand.write(reinterpret_cast<char*>(&last_move_buffer[0]), step * sizeof(int));

      bestand.close();

      //just val
      ifstream inlezen;
      int new_step;
      inlezen.open("./data/data_"+str(id), ios::in | ios::binary);
      inlezen.read(reinterpret_cast<char*>(&new_step), sizeof(int));

      board_buff_type new_board_buffer(new_step, vector<vector<int>>(BORAD_SIZE, vector<int>(BORAD_SIZE)));
      p_buff_type new_p_buffer(new_step, vector<float>(BORAD_SIZE * BORAD_SIZE));
      v_buff_type new_v_buffer(new_step);

      for (int i = 0; i < step; i++) {
          for (int j = 0; j < BORAD_SIZE; j++) {
              inlezen.read(reinterpret_cast<char*>(&new_board_buffer[i][j][0]), BORAD_SIZE * sizeof(int));
          }
      }

      for (int i = 0; i < step; i++) {
          inlezen.read(reinterpret_cast<char*>(&new_p_buffer[i][0]), BORAD_SIZE * BORAD_SIZE * sizeof(float));
      }

      inlezen.read(reinterpret_cast<char*>(&new_v_buffer[0]), step * sizeof(int));
  }


  //return { *board_buffer , *p_buffer , *cur_color_buff_type };

void SelfPlay::self_play_for_train(unsigned int game_num,unsigned int start_batch_id){
    std::vector<std::future<void>> futures;
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(&SelfPlay::play, this, start_batch_id+i));
        futures.emplace_back(std::move(future));
    }
    this->nn->batch_size = game_num * NUM_MCT_THREADS;
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
       
        this->nn->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
        cout << "end" << endl;
    }
    //return { *this->board_buffer , *this->p_buffer ,*this->v_buffer };
}
