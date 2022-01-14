#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 16:37:38 2021

@author: vittorio
"""

import os
import pickle

if not os.path.exists("./specs"):
    os.makedirs("./specs")

def save_obj(obj, name):
    with open('specs/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('specs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

environments = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2',
                'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3',
                'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1', 'HandManipulateBlock-v0',
                'HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandReach-v0']

specs = {}

specs['BipedalWalker-v3']={}
specs['BipedalWalker-v3']['number_steps_per_iter'] = 30000
specs['BipedalWalker-v3']['eval_freq'] = 1
specs['BipedalWalker-v3']['max_iter'] = 200
specs['BipedalWalker-v3']['evaluation_episodes'] = 10
specs['BipedalWalker-v3']['evaluation_max_n_steps'] = 2000

specs['BipedalWalkerHardcore-v3']={}
specs['BipedalWalkerHardcore-v3']['number_steps_per_iter'] = 30000
specs['BipedalWalkerHardcore-v3']['eval_freq'] = 1
specs['BipedalWalkerHardcore-v3']['max_iter'] = 334
specs['BipedalWalkerHardcore-v3']['evaluation_episodes'] = 10
specs['BipedalWalkerHardcore-v3']['evaluation_max_n_steps'] = 2000

specs['LunarLander-v2']={}
specs['LunarLander-v2']['number_steps_per_iter'] = 30000
specs['LunarLander-v2']['eval_freq'] = 1
specs['LunarLander-v2']['max_iter'] = 200
specs['LunarLander-v2']['evaluation_episodes'] = 10
specs['LunarLander-v2']['evaluation_max_n_steps'] = 2000

specs['LunarLanderContinuous-v2']={}
specs['LunarLanderContinuous-v2']['number_steps_per_iter'] = 30000
specs['LunarLanderContinuous-v2']['eval_freq'] = 1
specs['LunarLanderContinuous-v2']['max_iter'] = 200
specs['LunarLanderContinuous-v2']['evaluation_episodes'] = 10
specs['LunarLanderContinuous-v2']['evaluation_max_n_steps'] = 2000

specs['Ant-v3']={}
specs['Ant-v3']['number_steps_per_iter'] = 30000
specs['Ant-v3']['eval_freq'] = 1
specs['Ant-v3']['max_iter'] = 200
specs['Ant-v3']['evaluation_episodes'] = 10
specs['Ant-v3']['evaluation_max_n_steps'] = 2000

specs['HalfCheetah-v3']={}
specs['HalfCheetah-v3']['number_steps_per_iter'] = 30000
specs['HalfCheetah-v3']['eval_freq'] = 1
specs['HalfCheetah-v3']['max_iter'] = 200
specs['HalfCheetah-v3']['evaluation_episodes'] = 10
specs['HalfCheetah-v3']['evaluation_max_n_steps'] = 2000

specs['Hopper-v3']={}
specs['Hopper-v3']['number_steps_per_iter'] = 30000
specs['Hopper-v3']['eval_freq'] = 1
specs['Hopper-v3']['max_iter'] = 200
specs['Hopper-v3']['evaluation_episodes'] = 10
specs['Hopper-v3']['evaluation_max_n_steps'] = 2000

specs['Humanoid-v3']={}
specs['Humanoid-v3']['number_steps_per_iter'] = 30000
specs['Humanoid-v3']['eval_freq'] = 1
specs['Humanoid-v3']['max_iter'] = 200
specs['Humanoid-v3']['evaluation_episodes'] = 10
specs['Humanoid-v3']['evaluation_max_n_steps'] = 2000

specs['HumanoidStandup-v2']={}
specs['HumanoidStandup-v2']['number_steps_per_iter'] = 30000
specs['HumanoidStandup-v2']['eval_freq'] = 1
specs['HumanoidStandup-v2']['max_iter'] = 200
specs['HumanoidStandup-v2']['evaluation_episodes'] = 10
specs['HumanoidStandup-v2']['evaluation_max_n_steps'] = 2000

specs['Swimmer-v3']={}
specs['Swimmer-v3']['number_steps_per_iter'] = 30000
specs['Swimmer-v3']['eval_freq'] = 1
specs['Swimmer-v3']['max_iter'] = 200
specs['Swimmer-v3']['evaluation_episodes'] = 10
specs['Swimmer-v3']['evaluation_max_n_steps'] = 2000

specs['Walker2d-v3']={}
specs['Walker2d-v3']['number_steps_per_iter'] = 30000
specs['Walker2d-v3']['eval_freq'] = 1
specs['Walker2d-v3']['max_iter'] = 200
specs['Walker2d-v3']['evaluation_episodes'] = 10
specs['Walker2d-v3']['evaluation_max_n_steps'] = 2000

save_obj(specs, 'specs')

# %%




