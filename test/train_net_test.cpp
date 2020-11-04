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
    NeuralNetwork* model = new NeuralNetwork("./weights/" + str(current_weight) + ".pt", NUM_MCT_THREADS * NUM_MCT_SIMS);
    SelfPlay* sp = new SelfPlay(model);
    sp->self_play_for_train(NUM_TRAIN_THREADS, start_batch_id);
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
    v_buff_type v_buffer(8*sum_steps);
    

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


    vector<Tensor> board_tensor_buffer(sum_steps * 8);
    vector<Tensor> p_tensor_buffer(sum_steps * 8);

    for (int i = 0; i < sum_steps; i++) {
        board_tensor_buffer[i] = NeuralNetwork::transorm_board_to_Tensor(board_buffer[i], last_move_buffer[i], col_buffer[i]);
        p_tensor_buffer[i] = 
            torch::from_blob(&p_buffer[i][0], { 1, BORAD_SIZE,BORAD_SIZE }, torch::dtype(torch::kFloat32));

        board_tensor_buffer[i + sum_steps] = torch::rot90(board_tensor_buffer[i],/*k=*/1,/*dims=*/{ 2,3 });
        p_tensor_buffer[i+ sum_steps] = torch::rot90(p_tensor_buffer[i],/*k=*/1,/*dims=*/{ 1,2 });

        board_tensor_buffer[i + 2 * sum_steps] = torch::rot90(board_tensor_buffer[i],/*k=*/2,/*dims=*/{ 2,3 });
        p_tensor_buffer[i+2* sum_steps] = torch::rot90(p_tensor_buffer[i],/*k=*/2,/*dims=*/{ 1,2 });

        board_tensor_buffer[i + 3 * sum_steps] = torch::rot90(board_tensor_buffer[i],/*k=*/3,/*dims=*/{ 2,3 });
        p_tensor_buffer[i+3* sum_steps] = torch::rot90(p_tensor_buffer[i],/*k=*/3,/*dims=*/{ 1,2 });
    }

    for (int i = 0; i < 4*sum_steps; i++) {
        board_tensor_buffer[i + 4*sum_steps] = torch::flip(board_tensor_buffer[i],/*dims=*/{ 0 });
        p_tensor_buffer[i + 4 * sum_steps] = torch::flip(p_tensor_buffer[i],/*dims=*/{ 0 });
    }

    for (int i = 0; i < sum_steps; i++) {
        for (int j = 1; j < 8; j++) {
            v_buffer[i + j * sum_steps] = v_buffer[i];
        }
    }

    //for (int i = 0; i < v_buffer.size(); i++)
    //    cout << v_buffer.at(i) << endl;

    //delete &board_buffer;
    //delete &last_move_buffer;
   //delete &col_buffer;
    //

    unsigned seed = rand();
    //std::chrono::system_clock::now().time_since_epoch().count();
    auto e = std::default_random_engine(seed);
    shuffle(board_tensor_buffer.begin(), board_tensor_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(p_tensor_buffer.begin(), p_tensor_buffer.end(), e);
    e = std::default_random_engine(seed);
    shuffle(v_buffer.begin(), v_buffer.end(), e);

    //cout << board_tensor_buffer[1] << endl;
    //cout << p_tensor_buffer[1] << endl;

    model->train();
    torch::AutoGradMode enable_grad(true);
    for (int pt = 0; pt < sum_steps*8 - BATCH_SIZE; pt += BATCH_SIZE) {
        //std::cout << "train pt = " << pt << std::endl;
        Tensor inputs = cat(vector<Tensor>(board_tensor_buffer.begin() + pt, board_tensor_buffer.begin() + pt + BATCH_SIZE), 0);
        Tensor ps = cat(vector<Tensor>(p_tensor_buffer.begin() + pt, p_tensor_buffer.begin() + pt + BATCH_SIZE), 0).view({ -1, BORAD_SIZE * BORAD_SIZE });
        Tensor vs = torch::tensor(v_buff_type(v_buffer.begin() + pt, v_buffer.begin() + pt + BATCH_SIZE)).unsqueeze(1);

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

void play_for_eval(NeuralNetwork* a, NeuralNetwork* b, bool a_first, int* win_table, bool do_render,const int a_mct_sims,const int b_mct_sims) {
    MCTS ma(a, NUM_MCT_THREADS, C_PUCT, a_mct_sims, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    MCTS mb(b, NUM_MCT_THREADS, C_PUCT, b_mct_sims, C_VIRTUAL_LOSS, BORAD_SIZE * BORAD_SIZE);
    int step = 0;
    auto g = std::make_shared<Gomoku>(BORAD_SIZE, N_IN_ROW, BLACK);
    std::pair<int, int> game_state = g->get_game_status();
    //std::cout << episode << " th game!!" << std::endl;
    while (game_state.first == 0) {
        int res = (step + a_first) % 2 ? ma.get_best_action(g.get()) : mb.get_best_action(g.get());
        ma.update_with_move(res);
        mb.update_with_move(res);
        g->execute_move(res);
        if(do_render) g->render();
        game_state = g->get_game_status();
        step++;
    }
    cout << "eval: total step num = " << step << endl;

    if ((game_state.second == BLACK && a_first) || (game_state.second == WHITE && !a_first)) {
        cout << "winner = a" << endl;
        win_table[0]++;
    }
    else if ((game_state.second == BLACK && !a_first) || (game_state.second == WHITE && a_first)) {
        cout << "winner = b" << endl;
        win_table[1]++;
    } 
    else if (game_state.second == 0) win_table[2]++;
}



vector<int> eval(int weight_a, int weight_b, unsigned int game_num,int a_sims,int b_sims) {
    int win_table[3] = { 0,0,0 };
    
    ThreadPool *thread_pool = new ThreadPool(game_num);
    NeuralNetwork* nn_a = nullptr;
    NeuralNetwork* nn_b = nullptr;
    
    if (weight_a >= 0) {
        nn_a = new NeuralNetwork("./weights/" + str(weight_a) + ".pt", game_num * a_sims);
        cout << "NeuralNetwork A load: " << weight_a << endl;
    }
    else {
        cout << "NeuralNetwork A applies random policy!" << endl;
    }

    if (weight_b >= 0) {
        nn_b = new NeuralNetwork("./weights/" + str(weight_b) + ".pt", game_num * b_sims);
        cout << "NeuralNetwork B load: " << weight_b << endl;
    }
    else {
        cout << "NeuralNetwork B applies random policy!" << endl;
    }

    std::vector<std::future<void>> futures;
    //NeuralNetwork* a = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
    for (unsigned int i = 0; i < game_num; i++) {
        auto future = thread_pool->commit(std::bind(play_for_eval, nn_a, nn_b, i % 2 == 0, win_table,i==0, a_sims, b_sims));
        futures.emplace_back(std::move(future));
    }
    for (unsigned int i = 0; i < futures.size(); i++) {
        futures[i].wait();
        if (nn_a != nullptr){
            nn_a->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
        }
        if (nn_b != nullptr){
            nn_b->batch_size = max((unsigned)1, (game_num - i) * NUM_MCT_THREADS);
        }
        
    }
    //cout << "win_table = " << win_table[0] << win_table[1] << win_table [2] << endl;

    return { win_table[0], win_table[1],win_table[2] };
}

//std::pair<int, int> eval(int current_weight, int best_weight) {
//    int a_win_count = 0;
//    int b_win_count = 0;
//    //int tie = 0;
//    NeuralNetwork* a = new NeuralNetwork(NUM_MCT_THREADS);
//    NeuralNetwork* b = new NeuralNetwork(NUM_MCT_THREADS);
//    a->load_weights("./weights/" + str(current_weight) + ".pt");
//    b->load_weights("./weights/" + str(best_weight) + ".pt");
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
//            g->render();
//            game_state = g->get_game_status();
//            step++;
//        }
//        cout << "eval: total step num = " << step;
//        if ((game_state.second == BLACK && episode % 2 == 0) || (game_state.second == WHITE && episode % 2 == 1)) {
//            cout << "  winner = current_weight" << endl;
//            a_win_count++;
//        }
//        else if ((game_state.second == BLACK && episode % 2 == 1) || (game_state.second == WHITE && episode % 2 == 0)) {
//            cout << "  winner = old_best_weight" << endl;
//            b_win_count++;
//        } 
//        //else if (game_state.second == 0) tie++;
//#ifdef USE_GPU
//        c10::cuda::CUDACachingAllocator::emptyCache();
//#endif
//
//    }
//
//    return { a_win_count ,b_win_count };
//}


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
#ifdef JIT_MODE
            cout << "generate initial weight by python" << endl;
            //system("python ..\\data\\src\\learner.py train");
#else
            NeuralNetwork* model = new NeuralNetwork(NUM_MCT_THREADS * NUM_MCT_SIMS);
            model->save_weights("./weights/0.pt");
#endif
        ofstream weight_logger_writer("current_and_best_weight.txt");
        weight_logger_writer << 0 << " " << 0;
        weight_logger_writer.close();

        ofstream random_mcts_logger_writer("random_mcts_number.txt");
        random_mcts_logger_writer << NUM_MCT_SIMS;
        random_mcts_logger_writer.close();
    }else if (strcmp(argv[1], "generate") == 0) {
        cout << "generate " << atoi(argv[2])  << "-th batch."<< endl;
        int current_weight;
        //int best_weight;

        ifstream logger_reader("current_and_best_weight.txt");
        logger_reader >> current_weight;
        //logger_reader >> best_weight;
        if (current_weight < 0) {
            cout << "LOAD error,check current_and_best_weight.txt" << endl;
            return -1;
        }
        // logger_reader >> temp[1];
        cout << "Generating... current_weight = " << current_weight << endl;
        logger_reader.close();
        generate_data_for_train(current_weight, atoi(argv[2]) * NUM_TRAIN_THREADS);
    }
    else if (strcmp(argv[1], "train") == 0) {
        int current_weight;
        int best_weight;

        ifstream weight_logger_reader("current_and_best_weight.txt");
        weight_logger_reader >> current_weight;
        weight_logger_reader >> best_weight;
        cout << "Training... current_weight = " << current_weight << " and best_weight = " << best_weight << endl;
        

        weight_logger_reader.close();
        train(current_weight, atoi(argv[2]) * NUM_TRAIN_THREADS);
        current_weight++;

        ofstream weight_logger_writer("current_and_best_weight.txt");
        weight_logger_writer << current_weight << " " << best_weight;
        weight_logger_writer.close();
    }
	else if (strcmp(argv[1], "eval") == 0) {
		int current_weight;
        int best_weight;

        ifstream weight_logger_reader("current_and_best_weight.txt");
        weight_logger_reader >> current_weight;
        weight_logger_reader >> best_weight;
        cout << "Evaluating... current_weight = " << current_weight << " and best_weight = " << best_weight << endl;

        int game_num = atoi(argv[2]);


        auto result = eval(current_weight, best_weight, game_num, NUM_MCT_SIMS, NUM_MCT_SIMS);
        string result_log_info = str(current_weight) + "-th weight win: " + str(result[0]) + "  " + str(best_weight) + "-th weight win: " + str(result[1]) + "  tie: " + str(result[2]) + "\n";
        

        float win_ratio = result[0] / (result[1]+0.01);
        if (win_ratio > 1.2 ) {
            result_log_info += "new best weight: " + str(current_weight) + " generated!!!!\n";
            ofstream weight_logger_writer("current_and_best_weight.txt");
            weight_logger_writer << current_weight << " " << current_weight;
            weight_logger_writer.close();
        }
        cout << result_log_info;

        int random_mcts_simulation = NULL;
        ifstream random_mcts_logger_reader("random_mcts_number.txt");
        random_mcts_logger_reader >> random_mcts_simulation;

        int nn_mcts_simulation = NUM_MCT_SIMS / 4;

        vector<int> result_random_mcts = eval(current_weight, -1, game_num, nn_mcts_simulation, random_mcts_simulation);
        string result_log_info2 = str(current_weight) + "-th weight with mcts ["+ str(nn_mcts_simulation) + "] win: " + str(result_random_mcts[0]) + "  Random mcts ["+str(random_mcts_simulation)+ "] win: " + str(result_random_mcts[1]) + "  tie: " + str(result_random_mcts[2]) + "\n";
        if (result_random_mcts[0] - game_num==0 && random_mcts_simulation < 5000) {
            random_mcts_simulation += 100;
            result_log_info2 += "add random mcts number to: " + str(random_mcts_simulation) + "\n";

            ofstream random_mcts_logger_writer("random_mcts_number.txt");
            random_mcts_logger_writer << random_mcts_simulation;
            random_mcts_logger_writer.close();
        }
        cout << result_log_info2;

        ofstream detail_logger_writer("logger.txt", ios::app);
        detail_logger_writer << result_log_info << result_log_info2;
        detail_logger_writer.close();
    }


    //  #ifdef USE_GPU
    //      c10::cuda::CUDACachingAllocator::emptyCache();
    //  #endif
    //  }
}
