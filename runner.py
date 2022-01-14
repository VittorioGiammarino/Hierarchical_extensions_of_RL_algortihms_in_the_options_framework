#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:42:55 2021

@author: vittorio
"""

import World 
import numpy as np

from HVI import H_Value_iteration
from utils import Plot_summary

from evaluation import evaluate_H
from evaluation import eval_policy

class run_HVI:
    def __init__(self, env):
        self.env = env
        
    def run(self, plot = False):
        
        if self.env == "twoRooms":

            Environment = World.TwoRooms.Environment()
            stateSpace = Environment.stateSpace
            nActions = Environment.action_size
            
            nOptions = 2
            reward = np.zeros((len(stateSpace), nOptions, nActions))
            reward[25, 0, 2] = 1
            reward[50, 1, 4] = 1
            sum_rewards = np.sum(reward)
            reward = reward/sum_rewards
            
            pi_lo, pi_hi, pi_b, values_state_option = H_Value_iteration(self.env, Environment, nOptions, reward)
            
            if plot:
                Plot_summary(self.env, Environment, pi_lo, pi_hi, pi_b, values_state_option, (15,6))
                
        elif self.env == "twoRooms_degenerate":
            
            Environment = World.TwoRooms.Environment()
            stateSpace = Environment.stateSpace
            nActions = Environment.action_size
            
            nOptions = 2
            reward = np.zeros((len(stateSpace), nOptions, nActions))
            reward[50, 0, 4] = 1
            reward[50, 1, 4] = 1
            sum_rewards = np.sum(reward)
            reward = reward/sum_rewards
            
            pi_lo, pi_hi, pi_b, values_state_option = H_Value_iteration(self.env, Environment, nOptions, reward)
            if plot:
                Plot_summary(self.env, Environment, pi_lo, pi_hi, pi_b, values_state_option, (15,6))

        elif self.env == "twoRooms_flattened":
            
            Environment = World.TwoRooms.Environment()
            stateSpace = Environment.stateSpace
            nActions = Environment.action_size
            
            nOptions = 2
            reward = np.zeros((len(stateSpace), nOptions, nActions))
            reward[25, 0, 2] = 1
            reward[50, 1, 4] = 2
            sum_rewards = np.sum(reward)
            reward = reward/sum_rewards
            
            pi_lo, pi_hi, pi_b, values_state_option = H_Value_iteration(self.env, Environment, nOptions, reward)
            if plot:
                Plot_summary(self.env, Environment, pi_lo, pi_hi, pi_b, values_state_option, (15,6))

        elif self.env == "fourRooms":
            
            Environment = World.Four_Rooms.Environment()
            stateSpace = Environment.stateSpace
            nActions = Environment.action_size
            
            nOptions = 2
            reward = np.zeros((len(stateSpace), nOptions, nActions))
            reward[15, 0, 2] = 1
            reward[47, 0, 0] = 1
            reward[103, 1, 4] = 2
            sum_rewards = np.sum(reward)
            reward = reward/sum_rewards
            
            pi_lo, pi_hi, pi_b, values_state_option = H_Value_iteration(self.env, Environment, nOptions, reward)
            if plot:
                Plot_summary(self.env, Environment, pi_lo, pi_hi, pi_b, values_state_option, (15,8))

        elif self.env == "fourRooms_4Rewards":
            
            Environment = World.Four_Rooms.Environment()
            stateSpace = Environment.stateSpace
            nActions = Environment.action_size
            
            nOptions = 2
            reward = np.zeros((len(stateSpace), nOptions, nActions))
            reward[9, 0, 4] = 1
            reward[15, 0, 2] = 1
            reward[47, 0, 0] = 1
            reward[94, 0, 4] = 1
            reward[9, 1, 4] = 1
            reward[78, 1, 3] = 1
            reward[46, 1, 1] = 1
            reward[94, 1, 4] = 1
            
            pi_lo, pi_hi, pi_b, values_state_option = H_Value_iteration(self.env, Environment, nOptions, reward, 0.01)
            if plot:
                Plot_summary(self.env, Environment, pi_lo, pi_hi, pi_b, values_state_option, (15,8))
                
        else:
            print("Experiment not available, customization needed")
            
class run_SAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(np.array(state))

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            self.agent.Buffer.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train()

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent
        
class run_TD3:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample() 
                elif args.action_space == "Discrete":
                    action = env.action_space.sample()  
            else:
                action = self.agent.explore(state, args.expl_noise)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            self.agent.Buffer.add(state, action, next_state, reward, 0, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train()

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 

            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent     
    
class run_PPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent 
    
class run_TRPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent 
    
class run_UATRPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent 
    
class run_GePPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_RL = []
        avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_RL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Generate_and_store_rollout(env)
            self.agent.ADV_trace()
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = eval_policy(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_RL.append(avg_reward)    
                 
        return evaluation_RL, self.agent 
    
class run_HSAC:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        
        initial_option = 0
        initial_b = 1
        
        option = self.agent.select_option(state, initial_b, initial_option)

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.select_action(np.array(state), option)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            
            termination = self.agent.select_termination(next_state, option)
            
            if termination == 1:
                cost = self.agent.eta
            else:
                cost = 0

            self.agent.Buffer[option].add(state, action, next_state, reward, cost, done_bool)
            
            next_option = self.agent.select_option(next_state, termination, option)

            state = next_state
            option = next_option
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train(option)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                initial_option = 0
                initial_b = 1
                option = self.agent.select_option(state, initial_b, initial_option)
                
            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent
        
class run_HTD3:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 1
        
        initial_option = 0
        initial_b = 1
        
        option = self.agent.select_option(state, initial_b, initial_option)

        for t in range(int(args.number_steps_per_iter*args.max_iter)):
		
            episode_timesteps += 1

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                if args.action_space == "Continuous":
                    action = env.action_space.sample()
                elif args.action_space == "Discrete":
                    action = env.action_space.sample() 
            else:
                action = self.agent.explore(state, option, args.expl_noise)

            # Perform action
            next_state, reward, done, _ = env.step(action) 
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
            
            termination = self.agent.select_termination(next_state, option)
            
            if termination == 1:
                cost = self.agent.eta
            else:
                cost = 0

            self.agent.Buffer[option].add(state, action, next_state, reward, cost, done_bool)
            
            next_option = self.agent.select_option(next_state, termination, option)

            state = next_state
            option = next_option
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                self.agent.train(option)

            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                # Reset environment
                print(f"Total T: {t+1} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                state, done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                initial_option = 0
                initial_b = 1
                option = self.agent.select_option(state, initial_b, initial_option)
                
            # Evaluate episode
            if (t + 1) % (args.eval_freq*args.number_steps_per_iter) == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent    
    
class run_HPPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _, _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent    
    
class run_HTRPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _, _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent  
    
class run_HUATRPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            _, _, _, _ = self.agent.GAE(env)
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent  

class run_HGePPO:
    def __init__(self, agent):
        self.agent = agent
        
    def run(self, env, args, seed):
                # Evaluate untrained policy
        evaluation_HRL = []
        avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
        evaluation_HRL.append(avg_reward) 
    
        for i in range(int(args.max_iter)):
                        
            self.agent.Generate_and_store_rollout(env)
            self.agent.H_ADV_trace()
            self.agent.train(Entropy = True) 

            # Evaluate episode
            if (i + 1) % args.eval_freq == 0:
                avg_reward = evaluate_H(seed, self.agent, args.env, args.evaluation_max_n_steps, args.evaluation_episodes)
                evaluation_HRL.append(avg_reward)    
                 
        return evaluation_HRL, self.agent   
