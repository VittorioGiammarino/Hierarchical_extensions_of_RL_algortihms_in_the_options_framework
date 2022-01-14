#!/bin/bash

for seed in $(seq 0 9);
do
qsub SAC.qsub $seed 
qsub TD3.qsub $seed 
qsub PPO.qsub $seed 
qsub TRPO.qsub $seed 
qsub UATRPO.qsub $seed
qsub GePPO.qsub $seed 
done 
