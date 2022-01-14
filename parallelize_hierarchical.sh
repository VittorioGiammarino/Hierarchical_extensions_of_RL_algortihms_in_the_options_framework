#!/bin/bash

for seed in $(seq 0 9);
do
for option in $(seq 2 3);
do 
qsub H_PPO.qsub $seed $option
qsub H_GePPO.qsub $seed $option
qsub H_TRPO.qsub $seed $option
qsub H_UATRPO.qsub $seed $option
qsub H_SAC.qsub $seed $option
qsub H_TD3.qsub $seed $option
done 
done
