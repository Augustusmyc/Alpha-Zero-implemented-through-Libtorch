:: cmd /k "train_net_test.exe prepare"

set n=1

:loop
echo %n%-th train
call train_net_test.exe generate 0
call train_net_test.exe train 1
call train_net_test.exe eval 10
set n=n+1
goto loop