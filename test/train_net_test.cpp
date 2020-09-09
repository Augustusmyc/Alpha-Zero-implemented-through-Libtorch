#include <iostream>

#include <mcts.h>

#include <torch/script.h>
#include<torch/torch.h>
#include <libtorch.h>
#include <play.h>
#include <common.h>

#ifdef USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#endif

using namespace std;
using namespace torch;

static inline Tensor alpha_loss(Tensor& log_ps, Tensor& vs, const Tensor& target_ps, const Tensor& target_vs) {
    return mean(pow(vs - target_vs, 2)) - mean(sum(target_ps * log_ps, 1)); // value_loss + policy_loss
}

void generate_data_for_train(int current_weight, int start_batch_id) {
    NeuralNetwork* model = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
    model->load_weights("./weights/"+str(current_weight)+".pt");
    SelfPlay* sp = new SelfPlay(model);
    sp->self_play_for_train(NUM_TRAIN_THREADS, start_batch_id);
}


Tensor transorm_board_to_Tensor(board_type board, int last_move, int cur_player) {
    std::vector<int> board0;
    for (unsigned int i = 0; i < BORAD_SIZE; i++) {
        board0.insert(board0.end(), board[i].begin(), board[i].end());
    }

    torch::Tensor temp =
        torch::from_blob(&board0[0], { 1, 1, BORAD_SIZE, BORAD_SIZE}, torch::dtype(torch::kInt32));

    torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
    torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

    if (cur_player == -1) {
        std::swap(state0, state1);
    }

    torch::Tensor state2 =
        torch::zeros({ 1, 1, BORAD_SIZE, BORAD_SIZE }, torch::dtype(torch::kFloat32));

    if (last_move != -1) {
        state2[0][0][last_move / BORAD_SIZE][last_move % BORAD_SIZE] = 1;
    }

     torch::Tensor states = torch::cat({state0, state1}, 1);
    return cat({ state0, state1, state2 }, 1);
}

shared_ptr<AlphaZeroNet> train(int current_weight, int data_batch_num) {
    shared_ptr<AlphaZeroNet> model = make_shared<AlphaZeroNet>(
        AlphaZeroNet(/*num_layers=*/NUM_LAYERS,/*num_channels=*/NUM_CHANNELS,
            /*n=*/BORAD_SIZE,/*action_size=*/BORAD_SIZE * BORAD_SIZE));
    load(model, "./weights/" + str(current_weight) + ".pt");

    optim::Adam* optimizer = new optim::Adam(model->parameters(), optim::AdamOptions(LR));

    ifstream inlezen;
    vector<int> step_list(data_batch_num);
    for (int s_i = 0; s_i < data_batch_num; s_i++) {
        inlezen.open("./data/data_" + str(s_i), ios::in | ios::binary);
        inlezen.read(reinterpret_cast<char*>(&step_list[s_i]), sizeof(int));
        inlezen.close();
    }
    int sum_steps = 0;
    for (auto s : step_list) { sum_steps += s; }

    board_buff_type board_buffer(sum_steps, vector<vector<int>>(BORAD_SIZE, vector<int>(BORAD_SIZE)));
    p_buff_type p_buffer(sum_steps, vector<float>(BORAD_SIZE * BORAD_SIZE));
    v_buff_type v_buffer(sum_steps);
    vector<int> col_buffer(sum_steps);
    vector<int> last_move_buffer(sum_steps);

    int start_step = 0;
    for (int s_i = 0; s_i < data_batch_num; s_i++) {
        inlezen.open("./data/data_" + str(s_i), ios::in | ios::binary);
        inlezen.read(reinterpret_cast<char*>(&step_list[s_i]), sizeof(int)); // repeat just for skipping
        for (int i = 0; i < step_list[s_i]; i++) {
            for (int j = 0; j < BORAD_SIZE; j++) {
                inlezen.read(reinterpret_cast<char*>(&board_buffer[start_step + i][j][0]), BORAD_SIZE * sizeof(int));
            }
        }

        for (int i = 0; i < step_list[s_i]; i++) {
            inlezen.read(reinterpret_cast<char*>(&p_buffer[start_step + i][0]), BORAD_SIZE * BORAD_SIZE * sizeof(float));
        }

        inlezen.read(reinterpret_cast<char*>(&v_buffer[start_step]), step_list[s_i] * sizeof(int));

        inlezen.read(reinterpret_cast<char*>(&col_buffer[start_step]), step_list[s_i] * sizeof(int));
        inlezen.read(reinterpret_cast<char*>(&last_move_buffer[start_step]), step_list[s_i] * sizeof(int));

        start_step += step_list[s_i];
        inlezen.close();
    }

    vector<Tensor> board_tensor_buffer(sum_steps);
    for (int i = 0; i < sum_steps; i++) {
        board_tensor_buffer[i] = transorm_board_to_Tensor(board_buffer[i], last_move_buffer[i], col_buffer[i]);
    }

    //delete &board_buffer;
    //delete &last_move_buffer;
   //delete &col_buffer;

    unsigned seed = rand();
    //std::chrono::system_clock::now().time_since_epoch().count();
    auto e = std::default_random_engine(seed);
    shuffle(board_tensor_buffer.begin(), board_tensor_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(p_buffer.begin(), p_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(v_buffer.begin(), v_buffer.end(), e);
    //e = std::default_random_engine(seed);
    //shuffle(col_buffer.begin(), col_buffer.end(), e);
    //e = std::default_random_engine(seed);
    //shuffle(last_move_buffer.begin(), last_move_buffer.end(), e);

    std::vector<Tensor> p_tensor(BATCH_SIZE);
    model->train();
    torch::AutoGradMode enable_grad(true);
    for (int pt = 0; pt < sum_steps - BATCH_SIZE; pt += BATCH_SIZE) {
        //std::cout << "train pt = " << pt << std::endl;
        Tensor inputs = cat(vector<Tensor>(board_tensor_buffer.begin() + pt, board_tensor_buffer.begin() + pt + BATCH_SIZE), 0);

        //std::cout << "inputs dim = " << inputs.dim() << std::endl;
        //std::cout << "inputs size = " << inputs.size(0) << " " << inputs.size(1) << std::endl;

        Tensor vs = torch::tensor(v_buff_type(v_buffer.begin() + pt, v_buffer.begin() + pt + BATCH_SIZE)).unsqueeze(1);

        for (int i = 0; i < BATCH_SIZE; i++) {
            //v_tensor[i] = torch::tensor(v_buffer[pt + i]);
            p_tensor[i] = torch::tensor(p_buffer[pt + i]).unsqueeze(0);
        }
        //std::cout << "vector size = " << p_tensor.size() << std::endl;
        Tensor ps = cat(p_tensor, 0);
        //Tensor vs = cat(v_tensor, 0);
        optimizer->zero_grad();
#ifdef USE_GPU
        inputs = inputs.to(kCUDA);
        ps = ps.to(kCUDA);
        vs = vs.to(kCUDA);
#endif
        auto result = model->forward(inputs);


        //Tensor log_ps, Tensor vs, const Tensor target_ps, const Tensor target_vs
        //torch::binary_cross_entropy
        Tensor loss = alpha_loss(result.first, result.second, ps, vs);
        std::cout << "loss:" << loss.item() << std::endl;
        loss.backward();
        optimizer->step();
        //delete &inputs;
        //delete &ps;
        //delete &vs;
    }

    save(model, "./weights/" + str(current_weight+1) + ".pt");
    return model;
}


std::pair<int, int> eval(int current_weight, int best_weight) {
    int a_win_count = 0;
    int b_win_count = 0;
    //int tie = 0;
    NeuralNetwork* a = new NeuralNetwork(NUM_MCT_THREADS);
    NeuralNetwork* b = new NeuralNetwork(NUM_MCT_THREADS);
    a->load_weights("./weights/" + str(current_weight) + ".pt");
    b->load_weights("./weights/" + str(best_weight) + ".pt");
    MCTS ma(a, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    MCTS mb(b, NUM_MCT_THREADS, C_PUCT, NUM_MCT_SIMS, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    for (int episode = 0; episode < 4; episode++) {
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
            g->render();
            game_state = g->get_game_status();
            step++;
        }
        cout << "eval: total step num = " << step;
        if ((game_state.second == BLACK && episode % 2 == 0) || (game_state.second == WHITE && episode % 2 == 1)) {
            cout << "  winner = current_weight" << endl;
            a_win_count++;
        }
        else if ((game_state.second == BLACK && episode % 2 == 1) || (game_state.second == WHITE && episode % 2 == 0)) {
            cout << "  winner = old_best_weight" << endl;
            b_win_count++;
        } 
        //else if (game_state.second == 0) tie++;
#ifdef USE_GPU
        c10::cuda::CUDACachingAllocator::emptyCache();
#endif

    }

    return { a_win_count ,b_win_count };
}


int main(int argc, char* argv[]) {
    if (strcmp(argv[1], "prepare") == 0) {
        cout << "Prepare for training." << endl;
            // system("mkdir weights");
#ifdef _WIN32
            system("mkdir .\\weights");
            system("mkdir .\\data");
#elif __linux__
            system("mkdir ./weights");
            system("mkdir ./data");
#endif
            NeuralNetwork* model = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
            model->save_weights("./weights/0.pt");
        ofstream logger_writer("current_and_best_weight.txt");
        logger_writer << 0 << " " << 0;
        logger_writer.close();
    }else if (strcmp(argv[1], "generate") == 0) {
        cout << "generate " << atoi(argv[2])  << "-th batch."<< endl;
        int current_weight;
        int best_weight;

        ifstream logger_reader("current_and_best_weight.txt");
        logger_reader >> current_weight;
        logger_reader >> best_weight;
        if (current_weight < 0) {
            cout << "LOAD error,check current_and_best_weight.txt" << endl;
            return -1;
        }
        // logger_reader >> temp[1];
        cout << "Generating... current_weight = " << current_weight << " and best_weight = " << best_weight << endl;
        logger_reader.close();
        generate_data_for_train(current_weight, atoi(argv[2]) * NUM_TRAIN_THREADS);
    }
    else {
        int current_weight;
        int best_weight;

        ifstream logger_reader("current_and_best_weight.txt");
        logger_reader >> current_weight;
        logger_reader >> best_weight;
        // logger_reader >> temp[1];
        cout << "Training... current_weight = " << current_weight << " and best_weight = " << best_weight << endl;
        logger_reader.close();
        train(current_weight, stoi(str(argv[2]))*NUM_TRAIN_THREADS);
        current_weight++;
        std::pair<int, int> result = eval(current_weight, best_weight);
        if (result.first > result.second) {
            cout << "!!!!! new best id = " << current_weight << endl;
            best_weight = current_weight;
        }

        ofstream logger_writer("current_and_best_weight.txt");
        logger_writer << current_weight;
        logger_writer << " ";
        logger_writer << best_weight;
        logger_writer.close();
    }


    //  #ifdef USE_GPU
    //      c10::cuda::CUDACachingAllocator::emptyCache();
    //  #endif
    //  }
}
