#!/bin/bash
n=1
batch_num=1
jit_mode=1
do_prepare=1
if [ do_prepare ]
then
bash ./train_net.sh prepare
	if [ jit_mode ]
	then
		python ../src/learner.py
	fi
fi


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
	if [ jit_mode ]
	then
		python ../src/learner.py train
	else
		bash ./train_net.sh train $batch_num
	fi
	bash ./train_net.sh eval 10
	let n++
done