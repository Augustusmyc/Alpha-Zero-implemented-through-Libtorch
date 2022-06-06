# Alpha Zero implemented through Libtorch
# NOTE
This project stops update now.
New Project:
https://github.com/Augustusmyc/Alpha-Zero-Gomoku

# Supported Games
Currently Only Gomoku and Tic-Tac-Toe (Welcome other game implementions by githubers)


# Supported OS System
linux/Windows (tested on Ubuntu 16/18 and Windows 10)


# Supported Enviroment
Both GPU and CPU (GPU test on Tesla V100 and GTX 1080 / CPU test on Intel i7)


# Language
Mainly C++ (for speed!), except that model can be trained by either c++ or python (pytorch + libtorch).


# Dependence
gcc(linux, test version:7.5.0) or visual studio 19(windows)

cmake 3.18+

libtorch (debug version)

python + pytorch (optional)


# Installation(linux)
mkdir build

cd build

cmake ..

make


# Installation(windows)
make new direction named "build" 

cd .\build

cmake -A x64 ..

cmake --build . --config Debug

cd .\Debug

open .sln file through visual Studio 19 and generate


# Train
Linux: 

copy *.sh to ./build

cd ./build

bash train.sh


Windows: 

double click the windows train.bat


# Human play with AI
run mcts_test, for example in linux:

./mcts_test ./weights/1000.pt 1

Here 1(0) = AI play with black(white) pieces. 


# About this project
这是一个使用C++(libtorch) 进行推理的Alpha Zero训练框架。

目前游戏仅支持五子棋和井字棋，如果有小伙伴愿意提供其它棋类游戏的源码，欢迎来合并。

支持多线程蒙特卡洛树搜索,该部分和模型推理部分均由c++完成（为了加快速度，并且避开python GIL的坑）

模型训练部分：支持python(pytorch)训练 + torchscript推理（jit模式），
也支持使用c++(libtorch)训练，后者整个流程完全使用c++实现，不过目前来看效果并不理想，原因不明。因时间关系此项目不再更新。

注:最新项目改用onnxruntime推理 + pytorch训练，onnxruntime只需c++,这意味着只有训练时需要python. 事实上训练时用到pytorch的时间并不长，主要时间都花在左右互搏的推理过程中。

由于我这边主要用linux训练，windows相应的代码可能不会及时更新，需要修改一下才能用，特别是bat文件部分。
