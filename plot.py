#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:44:56 2021

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as ptch
import pickle

def load_obj(name):
    with open('specs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

specs = load_obj('specs')

# environments = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2',
#                 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3',
#                 'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1', 'HandManipulateBlock-v0',
#                 'HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandReach-v0']

environments = ['BipedalWalker-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 
                'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3']

modes = ['RL', 'HRL']
RL_algorithms = ['PPO', 'TRPO', 'UATRPO', 'GePPO', 'TD3', 'SAC']
HRL = ['HPPO', 'HTRPO', 'HUATRPO', 'HGePPO', 'HTD3', 'HSAC']

colors = {}

colors['PPO'] = 'tab:blue'
colors['TRPO'] = 'tab:orange'
colors['UATRPO'] = 'tab:pink'
colors['GePPO'] = 'lime'
colors['TD3'] = 'tab:red'
colors['SAC'] = 'tab:purple'
colors['HPPO_2'] = 'tab:brown'
colors['HTRPO_2'] = 'tab:green'
colors['HUATRPO_2'] = 'tab:gray'
colors['HGePPO_2'] = 'chocolate'
colors['HTD3_2'] = 'tab:olive'
colors['HSAC_2'] = 'tab:cyan'
colors['HPPO_3'] = 'lightcoral'
colors['HTRPO_3'] = 'fuchsia'
colors['HUATRPO_3'] = 'gold'
colors['HGePPO_3'] = 'magenta'
colors['HTD3_3'] = 'lightseagreen'
colors['HSAC_3'] = 'peru'


# %%
for env in environments:
    
    columns = 3
    rows = 2
    
    fig, ax = plt.subplots(rows, columns, figsize=(20,7))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    fig.suptitle(env, fontsize="xx-large")
    i = 0
    
    for k, ax_row in enumerate(ax):
        for j, axes in enumerate(ax_row):
            
            policy = RL_algorithms[i]
            
            RL = []
            HRL_2 = []
            HRL_3 = []
            
            for seed in range(10):
                try:
                    with open(f'results/HRL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                        RL.append(np.load(f, allow_pickle=True))  
                except:
                    print(f"{policy}_{env}_{seed} not found")
                    
                with open(f'results/HRL/evaluation_HRL_H{policy}_nOptions_2_{env}_{seed}.npy', 'rb') as f:
                    HRL_2.append(np.load(f, allow_pickle=True))   
                    
                with open(f'results/HRL/evaluation_HRL_H{policy}_nOptions_3_{env}_{seed}.npy', 'rb') as f:
                    HRL_3.append(np.load(f, allow_pickle=True)) 
                
            mean = np.mean(np.array(RL),0)
            steps = np.linspace(0,(specs[env]['max_iter']*specs[env]['number_steps_per_iter']),len(mean))
            std = np.std(np.array(RL),0)
            axes.plot(steps, mean, label=policy, c=colors[policy])
            axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
            
            mean = np.mean(np.array(HRL_2),0)
            std = np.std(np.array(HRL_2),0)
            axes.plot(steps, mean, label=f'H{policy} 2 options', c=colors[f'H{policy}_2'])
            axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_2'])
            
            mean = np.mean(np.array(HRL_3),0)
            std = np.std(np.array(HRL_3),0)
            axes.plot(steps, mean, label=f'H{policy} 3 options', c=colors[f'H{policy}_3'])
            axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[f'H{policy}_3'])
            
            axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)
            axes.set_xlabel('Steps')
            axes.set_ylabel('Reward')
            
            i+=1
            
    plt.savefig(f'Figures/{env}/{env}_comparison.pdf', format='pdf', bbox_inches='tight')
            
            
            
            
    
    
