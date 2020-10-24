import struct
import os
import numpy as np

import struct



BOARD_SIZE = 3
if __name__ == '__main__':
    filepath=r'E:\Projects\Alpha-Cpp\build\data\data_5'
    with open(filepath, 'rb') as binfile:#打开二进制文件
        size = os.path.getsize(filepath) #获得文件大小
        step = binfile.read(4)
        step = int().from_bytes(step, byteorder='little', signed=True)
        print(step)
        board = np.zeros((step,BOARD_SIZE*BOARD_SIZE))
        for i in range(step):
           for j in range(BOARD_SIZE*BOARD_SIZE):
               data = binfile.read(4)
               data = int().from_bytes(data, byteorder='little', signed=True)
               board[i][j] = data
        board = board.reshape((-1,BOARD_SIZE,BOARD_SIZE))
        print(board)

        p = np.zeros((step,BOARD_SIZE*BOARD_SIZE))
        for i in range(step):
            for j in range(BOARD_SIZE*BOARD_SIZE):
                data=binfile.read(4)
                data=struct.unpack('f', data)[0]
                p[i][j] = data
        p = p.reshape((-1,BOARD_SIZE,BOARD_SIZE))
        print(p)
        
        v = []
        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            v.append(data)
        print(v)

        color= []
        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            color.append(data)
        print(color)

        last_action = []

        for i in range(step):
            data = binfile.read(4)
            data = int().from_bytes(data, byteorder='little', signed=True)
            last_action.append(data)
        print(last_action)