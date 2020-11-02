cmake_minimum_required(VERSION 3.8.0)
project(cppgomu)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

# gcc
if(CMAKE_COMPILER_IS_GNUCC)
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/data/libtorch")
else()
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "D:/libtorch")
endif()

find_package(Torch REQUIRED)


# find python
#find_package(PythonLibs)
#include_directories(${PYTHON_INCLUDE_PATH})

# add sources
include_directories(./src)
aux_source_directory(./src SOURCES)

# unit test
add_library(test_lib ${SOURCES})
target_link_libraries(test_lib ${TORCH_LIBRARIES})

#add_executable(thread_pool_test ./test/thread_pool_test.cpp)
#target_link_libraries(thread_pool_test ${TORCH_LIBRARIES})

#add_executable(gomoku_test ./test/gomoku_test.cpp)
#target_link_libraries(gomoku_test test_lib)

#add_executable(libtorch_test ./test/libtorch_test.cpp)
#target_link_libraries(libtorch_test test_lib)

add_executable(mcts_test ./test/mcts_test.cpp)   # play game with human
target_link_libraries(mcts_test test_lib)

#add_executable(multi_games_test ./test/multi_games_test.cpp)
#target_link_libraries(multi_games_test test_lib)

add_executable(train_net_test ./test/train_net_test.cpp)
target_link_libraries(train_net_test test_lib)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET test_lib POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS} ${CMAKE_CURRENT_BINARY_DIR}/Debug)
endif (MSVC)
