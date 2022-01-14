#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:17:27 2021

@author: vittorio
"""

import torch
import argparse
import os
import numpy as np
import gym
from gym.wrappers import FilterObservation, FlattenObservation

import runner

import SAC
import TD3
import PPO
import TRPO
import UATRPO
import GePPO
import H_SAC
import H_TD3
import H_PPO
import H_TRPO
import H_UATRPO
import H_GePPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
    
def HRL(env, args, seed):
    
    if args.action_space == 'Continuous':
        action_dim = env.action_space.shape[0] 
        action_space_cardinality = np.inf
        max_action = np.zeros((action_dim,))
        min_action = np.zeros((action_dim,))
        for a in range(action_dim):
            max_action[a] = env.action_space.high[a]   
            min_action[a] = env.action_space.low[a]  
            
    elif args.action_space == 'Discrete':
        
        try:
            action_dim = env.action_space.shape[0] 
        except:
            action_dim = 1

        action_space_cardinality = env.action_space.n
        max_action = np.nan
        min_action = np.nan
                
    try:
        state_dim = env.observation_space.shape[0] 
    except:
        env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal', 'achieved_goal']))
        env.seed(seed)
        env.action_space.seed(seed)
        env._max_episode_steps = args.evaluation_max_n_steps
        state_dim = env.observation_space.shape[0]  

    option_dim = args.number_options
    termination_dim = 2
     
    if args.policy == "SAC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = SAC.SAC(**kwargs)
        
        run_sim = runner.run_SAC(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL
    
    if args.policy == "TD3":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action
        }

        Agent_RL = TD3.TD3(**kwargs)
        
        run_sim = runner.run_TD3(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL    
    
    if args.policy == "PPO":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_RL = PPO.PPO(**kwargs)
        
        run_sim = runner.run_PPO(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL  
    
    if args.policy == "TRPO":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "num_steps_per_rollout": args.number_steps_per_iter
        }
        
        Agent_RL = TRPO.TRPO(**kwargs)
        
        run_sim = runner.run_TRPO(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL  
    
    if args.policy == "UATRPO":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "num_steps_per_rollout": args.number_steps_per_iter
        }
        
        Agent_RL = UATRPO.UATRPO(**kwargs)
        
        run_sim = runner.run_UATRPO(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL  
    
    if args.policy == "GePPO":        
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_RL = GePPO.GePPO(**kwargs)
        
        run_sim = runner.run_GePPO(Agent_RL)
        evaluation_RL, Agent_RL = run_sim.run(env, args, seed)
        
        return evaluation_RL, Agent_RL 
    
    if args.policy == "HSAC":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim
        }

        Agent_HRL = H_SAC.H_SAC(**kwargs)
        
        run_sim = runner.run_HSAC(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "HTD3":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim
        }

        Agent_HRL = H_TD3.H_TD3(**kwargs)
        
        run_sim = runner.run_HTD3(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "HPPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_HRL = H_PPO.H_PPO(**kwargs)
        
        run_sim = runner.run_HPPO(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "HTRPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_HRL = H_TRPO.H_TRPO(**kwargs)
        
        run_sim = runner.run_HTRPO(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "HUATRPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_HRL = H_UATRPO.H_UATRPO(**kwargs)
        
        run_sim = runner.run_HUATRPO(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
    if args.policy == "HGePPO":
        kwargs = {
         "state_dim": state_dim,
         "action_dim": action_dim,
         "action_space_cardinality": action_space_cardinality,
         "max_action": max_action,
         "min_action": min_action,
         "option_dim": option_dim,
         "termination_dim": termination_dim,
         "num_steps_per_rollout": args.number_steps_per_iter
        }

        Agent_HRL = H_GePPO.H_GePPO(**kwargs)
        
        run_sim = runner.run_HGePPO(Agent_HRL)
        evaluation_HRL, Agent_HRL = run_sim.run(env, args, seed)
        
        return evaluation_HRL, Agent_HRL
    
def train(args, seed): 
    
    env = gym.make(args.env)
    
    try:
        if env.action_space.n>0:
            args.action_space = "Discrete"
            print("Environment supports Discrete action space.")
    except:
        args.action_space = "Continuous"
        print("Environment supports Continuous action space.")
            
    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    evaluations, policy = HRL(env, args, seed)
    
    return evaluations, policy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    #General
    parser.add_argument("--mode", default="RL", help='supported modes are HVI, HRL and RL (default = "HVI")')     
    parser.add_argument("--env", default="BipedalWalker-v3")               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--action_space", default="Continuous")               # Sets Gym, PyTorch and Numpy seeds
    
    parser.add_argument("--number_options", default=2, type=int)     # number of options
    parser.add_argument("--policy", default="GePPO")                   # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--number_steps_per_iter", default=30000, type=int) # Time steps initial random policy is used 25e3
    parser.add_argument("--eval_freq", default=1, type=int)          # How often (time steps) we evaluate
    parser.add_argument("--max_iter", default=200, type=int)    # Max time steps to run environment
    # HRL
    parser.add_argument("--start_timesteps", default=25e3, type=int) # Time steps before training default=25e3
    parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise    
    parser.add_argument("--save_model", action="store_false")         # Save model and optimizer parameters
    parser.add_argument("--load_model", default=True, type=bool)               # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model_path", default="") 
    # Evaluation
    parser.add_argument("--evaluation_episodes", default=10, type=int)
    parser.add_argument("--evaluation_max_n_steps", default = 2000, type=int)
    # Experiments
    parser.add_argument("--detect_gradient_anomaly", action="store_true")
    args = parser.parse_args()
    
    torch.autograd.set_detect_anomaly(args.detect_gradient_anomaly)
      
    if args.mode == "HVI":
        
        file_name = f"{args.mode}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Mode: {args.mode}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
        simulation = runner.run_HVI(args.env)
        simulation.run(True)
        
    if args.mode == "RL":
        
        file_name = f"{args.mode}_{args.policy}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
        if not os.path.exists("./results/HRL"):
            os.makedirs("./results/HRL")
            
        if not os.path.exists(f"./models/HRL/{file_name}"):
            os.makedirs(f"./models/HRL/{file_name}")
        
        
        evaluations, policy = train(args, args.seed)
        
        if args.save_model: 
            np.save(f"./results/HRL/evaluation_{file_name}", evaluations)
            policy.save_actor(f"./models/HRL/{file_name}/{file_name}")
            policy.save_critic(f"./models/HRL/{file_name}/{file_name}")
            
    if args.mode == "HRL":
        
        file_name = f"{args.mode}_{args.policy}_nOptions_{args.number_options}_{args.env}_{args.seed}"
        print("---------------------------------------")
        print(f"Mode: {args.mode}, Policy: {args.policy}, nOptions: {args.number_options}, Env: {args.env}, Seed: {args.seed}")
        print("---------------------------------------")
        
        if not os.path.exists("./results/HRL"):
            os.makedirs("./results/HRL")
            
        if not os.path.exists(f"./models/HRL/{file_name}"):
            os.makedirs(f"./models/HRL/{file_name}")
        
        
        evaluations, policy = train(args, args.seed)
        
        if args.save_model: 
            np.save(f"./results/HRL/evaluation_{file_name}", evaluations)
            policy.save_actor(f"./models/HRL/{file_name}/{file_name}")
            policy.save_critic(f"./models/HRL/{file_name}/{file_name}")
            
    if args.mode == "test":
            
        environments = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2',
                        'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3',
                        'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1', 'HandManipulateBlock-v0',
                        'HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandReach-v0']
        
        #policies = ['PPO', 'TRPO', 'UATRPO', 'TD3', 'SAC', 'H_PPO', 'H_TRPO', 'H_UATRPO', 'H_TD3', 'H_SAC']
        
        policies = ['HPPO', 'HTRPO', 'HUATRPO', 'HTD3', 'HSAC']
        
        args.number_steps_per_iter = 5000
        args.max_iter = 1
        args.evaluation_max_n_steps = 1000
        
        results = {}
        
        for pi in policies:
            results[pi]={}
            for environm in environments:
                
                args.env = environm
                args.policy = pi
        
                file_name = f"{args.mode}_{args.policy}_{args.env}_{args.seed}"
                print("---------------------------------------")
                print(f"Mode: {args.mode}, Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
                print("---------------------------------------")
        
                evaluations, policy = train(args, args.seed)
                
                results[args.policy][args.env] = evaluations
        
        
    
   
                
                
                