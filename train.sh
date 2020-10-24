#!/bin/bash
n=1
batch_num=1
bash ./train_net.sh prepare

while [ 1 ]
do
	echo "--------------$n-th train------------------"
	 for ((i=0;i<$batch_num;i++));do
		{
		# sleep 3;echo 1>>aa && echo "done!"
		bash ./train_net.sh generate $i
		}&
	done
	wait
	bash ./train_net.sh train $batch_num
	bash ./train_net.sh eval 10
	let n++
done