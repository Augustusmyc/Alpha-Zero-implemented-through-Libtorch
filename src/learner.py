from collections import deque
from os import path, mkdir
import os
# import threading
# import time
# import math
import numpy as np
# import pickle
# import concurrent.futures
import random, struct
from functools import reduce

import sys

sys.path.append('../build')

from neural_network import NeuralNetWorkWrapper


# def tuple_2d_to_numpy_2d(tuple_2d):
#     # help function
#     # convert type
#     res = [None] * len(tuple_2d)
#     for i, tuple_1d in enumerate(tuple_2d):
#         res[i] = list(tuple_1d)
#     return np.array(res)


class Learner():
    def __init__(self, config):
        # see config.py
        # gomoku
        self.n = config['n']
        self.n_in_row = config['n_in_row']
        # self.gomoku_gui = GomokuGUI(config['n'], config['human_color'])
        self.action_size = config['action_size']

        # train
        self.num_iters = config['num_iters']
        self.num_eps = config['num_eps']
        self.num_train_threads = config['num_train_threads']
        self.check_freq = config['check_freq']
        self.num_contest = config['num_contest']
        self.dirichlet_alpha = config['dirichlet_alpha']
        self.temp = config['temp']
        self.update_threshold = config['update_threshold']
        self.num_explore = config['num_explore']

        self.examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

        # mcts
        self.num_mcts_sims = config['num_mcts_sims']
        self.c_puct = config['c_puct']
        self.c_virtual_loss = config['c_virtual_loss']
        self.num_mcts_threads = config['num_mcts_threads']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # neural network
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.nnet = NeuralNetWorkWrapper(config['lr'], config['l2'], config['num_layers'],
                                         config['num_channels'], config['n'], self.action_size, config['train_use_gpu'],
                                         self.libtorch_use_gpu)

        # start gui
        # t = threading.Thread(target=self.gomoku_gui.loop)
        # t.start()

    def learn(self):
        # train the model by self play

        # model_id = 0
        # best_model = path.join('..','build','weights', str(model_id))
        # if path.exists(best_model+'.pkl'):
        #     print(f"loading {model_id}-th model")
        #     self.nnet.load_model(best_model)
        #     #self.load_samples()
        # else:
        #     print("prepare: save 0-th model")
        #     # save torchscript
        #     # self.nnet.save_model()
        #     self.nnet.save_model(best_model)

        data_path = path.join('..', 'build', 'data')
        train_data = self.load_samples(data_path)
        random.shuffle(train_data)

        # train neural network
        epochs = self.epochs * (len(train_data) + self.batch_size - 1) // self.batch_size
        self.nnet.train(train_data, self.batch_size, int(epochs))
        # self.nnet.save_model()

    def get_symmetries(self, board, pi, last_action):
        # mirror, rotational
        assert (len(pi) == self.action_size)  # 1 for pass

        pi_board = np.reshape(pi, (self.n, self.n))
        last_action_board = np.zeros((self.n, self.n))
        last_action_board[last_action // self.n][last_action % self.n] = 1
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                newAction = np.rot90(last_action_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                    newAction = np.fliplr(last_action_board)
                l += [(newB, newPi.ravel(), np.argmax(newAction) if last_action != -1 else -1)]
        return l

    def load_samples(self, folder):
        """load self.examples_buffer
        """
        BOARD_SIZE = self.n
        train_examples = []
        data_files = os.listdir(folder)
        for file_name in data_files:
            file_path = path.join(folder, file_name)
            with open(file_path, 'rb') as binfile:
                # size = os.path.getsize(filepath) #获得文件大小
                step = binfile.read(4)
                step = int().from_bytes(step, byteorder='little', signed=True)
                board = np.zeros((step, BOARD_SIZE * BOARD_SIZE))
                for i in range(step):
                    for j in range(BOARD_SIZE * BOARD_SIZE):
                        data = binfile.read(4)
                        data = int().from_bytes(data, byteorder='little', signed=True)
                        board[i][j] = data
                        # board = board.reshape((-1,BOARD_SIZE,BOARD_SIZE))
                prob = np.zeros((step, BOARD_SIZE * BOARD_SIZE))
                for i in range(step):
                    for j in range(BOARD_SIZE * BOARD_SIZE):
                        data = binfile.read(4)
                        data = struct.unpack('f', data)[0]
                        prob[i][j] = data
                        # p = p.reshape((-1,BOARD_SIZE,BOARD_SIZE))
                    # print(p)

                v = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    v.append(data)
                    # print(v)

                color = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    color.append(data)

                last_action = []
                for i in range(step):
                    data = binfile.read(4)
                    data = int().from_bytes(data, byteorder='little', signed=True)
                    last_action.append(data)

                for i in range(step):
                    sym = self.get_symmetries(board[i], prob[i], last_action[i])
                    for i, (b, p, a) in enumerate(sym):
                        train_examples.append([b, a, color[i], p, v[i]])
        return train_examples

    # def save_samples(self, folder="models", filename="checkpoint.example"):
    #     """save self.examples_buffer
    #     """

    #     if not path.exists(folder):
    #         mkdir(folder)

    #     filepath = path.join(folder, filename)
    #     with open(filepath, 'wb') as f:
    #         pickle.dump(self.examples_buffer, f, -1)


if __name__ == '__main__':
    import config

    le = Learner(config.config)
    le.learn()
