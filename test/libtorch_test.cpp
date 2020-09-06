#include <libtorch.h>

#include <iostream>
#include <gomoku.h>
#include <common.h>

int main() {
  Gomoku gomoku(BORAD_SIZE, N_IN_ROW, 1);

  // test execute_move
  gomoku.execute_move(3);
  gomoku.execute_move(4);
  gomoku.execute_move(6);

  // test render
  gomoku.render();

  std::cout << gomoku.get_last_move() << std::endl;
  std::cout << gomoku.get_current_color() << std::endl;

  NeuralNetwork nn(1);
  auto res = nn.commit(&gomoku).get();
  auto p = res[0];
  auto v = res[1];

  std::for_each(p.begin(), p.end(), [](double x) { std::cout << x << ","; });
  std::cout << std::endl;

  std::cout << v << std::endl;


  // 2
  gomoku.execute_move(2);
  std::cout << gomoku.get_last_move() << std::endl;
  std::cout << gomoku.get_current_color() << std::endl;

  res = nn.commit(&gomoku).get();
  p = res[0];
  v = res[1];

  std::for_each(p.begin(), p.end(), [](double x) { std::cout << x << ","; });
  std::cout << std::endl;

  std::cout << v << std::endl;

  // stress testing
  std::cout << "stress testing" << std::endl;
  auto start = std::chrono::system_clock::now();

  for (unsigned i = 0; i < 10; i++) {
    nn.commit(&gomoku);
  }

  res = nn.commit(&gomoku).get();
  auto end = std::chrono::system_clock::now();

  std::cout <<  "cost time:" << double(std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                       .count()) *
                   std::chrono::microseconds::period::num /
                   std::chrono::microseconds::period::den
            << std::endl;

  p = res[0];
  v = res[1];

  std::for_each(p.begin(), p.end(), [](double x) { std::cout << x << ","; });
  std::cout << std::endl;

  std::cout << v << std::endl;
}
