#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from models import TanhGaussianHierarchicalActor
from models import SoftmaxHierarchicalActor
from models import Value_net_H

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class H_PPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, option_dim, termination_dim, 
                 num_steps_per_rollout=5000, pi_b_freq=3, pi_hi_freq=3, l_rate_pi_lo=3e-4, l_rate_pi_hi=3e-4 , l_rate_pi_b=3e-4, l_rate_critic=3e-4, 
                 gae_gamma = 0.99, gae_lambda = 0.99, epsilon = 0.2, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs_pi_lo=5, num_epochs_pi_b=10, num_epochs_pi_hi=10, 
                 eta = 1e-7, min_batch_size = 64):
        
        self.pi_lo = [[None]*1 for _ in range(option_dim)]
        self.pi_b = [[None]*1 for _ in range(option_dim)]
        
        if np.isinf(action_space_cardinality):
            self.pi_hi = TanhGaussianHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            pi_lo_temp = TanhGaussianHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            pi_b_temp = TanhGaussianHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            self.action_space = "Continuous"
            
        else:
            self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            self.action_space = "Discrete"
            
        self.value_function = Value_net_H(state_dim, option_dim).to(device)
        
        for option in range(option_dim):
            self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b[option] = copy.deepcopy(pi_b_temp)  
            
        # define optimizer 
        self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=l_rate_pi_hi)
        self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.value_function_optimizer = torch.optim.Adam(self.value_function.parameters(), lr=l_rate_critic)
        for option in range(option_dim):
            self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=l_rate_pi_lo)
            self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=l_rate_pi_b)  
      
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        self.option_dim = option_dim
        self.termination_dim = termination_dim
        
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.minibatch_size = minibatch_size
        self.num_epochs_pi_hi = num_epochs_pi_hi
        self.num_epochs_pi_b = num_epochs_pi_b
        self.num_epochs_pi_lo = num_epochs_pi_lo
        self.eta = eta
        self.pi_b_freq = pi_b_freq
        self.pi_hi_freq = pi_hi_freq
        self.min_batch_size = min_batch_size
        
        self.Total_t = 0
        self.Total_iter = 0
        self.states = []
        self.actions = []
        self.options = []
        self.terminations = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
        self.advantage_option = []
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        
    def select_action(self, state, option):
        self.pi_lo[option].eval()
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            action, _ = self.pi_lo[option].sample(state)
            return int((action).cpu().data.numpy().flatten())
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            action, _, _ = self.pi_lo[option].sample(state)
            return (action).cpu().data.numpy().flatten()
        
    def select_option(self, state, b, previous_option):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)     
        if b == 1:
            b_bool = True
        else:
            b_bool = False

        o_prob_tilde = np.empty((1,self.option_dim))
        if b_bool == True:
            o_prob_tilde = self.pi_hi(state).cpu().data.numpy()
        else:
            o_prob_tilde[0,:] = 0
            o_prob_tilde[0,previous_option] = 1

        prob_o = torch.FloatTensor(o_prob_tilde)
        m = Categorical(prob_o)
        option = m.sample()
        
        return int(option.detach().data.numpy().flatten())
    
    def select_termination(self, state, option):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        termination, _ = self.pi_b[option].sample(state)
        return int((termination).cpu().data.numpy().flatten())
        
    def GAE(self, env):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.options = []
        self.terminations = []
        self.returns =  []
        self.advantage = []
        self.gammas = []
        
        self.advantage_option = []
        
        while step < self.num_steps_per_rollout: 
            episode_states = []
            episode_actions = []
            episode_options = []
            episode_terminations = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []   
            state, done = env.reset(), False
            t=0
            episode_reward = 0
            
            initial_option = 0
            initial_b = 1
            option = H_PPO.select_option(self, state, initial_b, initial_option)

            while not done and step < self.num_steps_per_rollout:            
                action = H_PPO.select_action(self, state, option)
            
                self.states.append(state)
                self.actions.append(action)
                self.options.append(option)
                episode_states.append(state)
                episode_actions.append(action)
                episode_options.append(option)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
                
                termination = H_PPO.select_termination(self, state, option)
                next_option = H_PPO.select_option(self, state, termination, option)
                option = next_option
                
                self.terminations.append(termination)
                episode_terminations.append(termination)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                
            episode_states = torch.FloatTensor(np.array(episode_states))
            episode_actions = torch.FloatTensor(np.array(episode_actions))
            episode_options = torch.LongTensor(np.array(episode_options))
            episode_terminations = torch.LongTensor(np.array(episode_terminations))
            episode_rewards = torch.FloatTensor(np.array(episode_rewards))
            episode_gammas = torch.FloatTensor(np.array(episode_gammas))
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas))         
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.returns.append(episode_returns)
            self.value_function.eval()
            self.pi_hi.eval()
            
            episode_option_vector = torch.ones_like(episode_options, dtype=int)
            episode_options = F.one_hot(episode_options, num_classes=self.option_dim)
            current_values = self.value_function(episode_states, episode_options).detach()
            next_values = torch.cat((self.value_function(episode_states, episode_options)[1:], torch.FloatTensor([[0.]]))).detach()
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            current_values_option = 0
            next_values_option = 0
            for option_i in range(self.option_dim):
                current_values_option_i = self.value_function(episode_states, F.one_hot(option_i*episode_option_vector, num_classes=self.option_dim)).detach()
                next_values_option_i = torch.cat((self.value_function(episode_states, F.one_hot(option_i*episode_option_vector, num_classes=self.option_dim))[1:], torch.FloatTensor([[0.]]))).detach()
                current_values_option += self.pi_hi(episode_states)[:,option_i].reshape(-1,1)*current_values_option_i
                next_values_option += torch.cat((self.pi_hi(episode_states)[1:,option_i].reshape(-1,1), torch.FloatTensor([[0.]])))*next_values_option_i
                    
            episode_deltas_option = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values_option - current_values_option
            episode_advantage_option = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas_option[j:]).sum() for j in range(t)])
             
            self.advantage.append(episode_advantage)
            self.advantage_option.append(episode_advantage_option)
            self.gammas.append(episode_gammas)
                
        rollout_states = torch.FloatTensor(np.array(self.states))
        rollout_actions = torch.FloatTensor(np.array(self.actions))
        rollout_options = torch.LongTensor(np.array(self.options))
        rollout_terminations = torch.LongTensor(np.array(self.terminations))

        return rollout_states, rollout_actions, rollout_options, rollout_terminations
    
    
    def train(self, Entropy = False):
        
        rollout_states = torch.FloatTensor(np.array(self.states))
        
        if self.action_space == "Discrete":
            rollout_actions = torch.LongTensor(np.array(self.actions))
        elif self.action_space == "Continuous":
            rollout_actions = torch.FloatTensor(np.array(self.actions))
        
        rollout_options = torch.LongTensor(np.array(self.options))
        rollout_terminations = torch.LongTensor(np.array(self.terminations))
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_advantage_option = torch.cat(self.advantage_option)      
        rollout_gammas = torch.cat(self.gammas)   
        
        index_op = []
        for option in range(self.option_dim):
            index_op.append(np.where(option==rollout_options)[0])
            
        for option in range(self.option_dim):
            
            states_op = rollout_states[index_op[option]]
            if len(states_op)>=self.min_batch_size:
                option_vector = rollout_options[index_op[option]]
                actions_op = rollout_actions[index_op[option]]
                returns_op = rollout_returns[index_op[option]]
                advantage_op = rollout_advantage[index_op[option]]
                gammas_op = rollout_gammas[index_op[option]]
                
                advantage_op = (advantage_op-advantage_op.mean())/(advantage_op.std()+1e-6)
                
                self.pi_lo[option].eval()
                
                if self.action_space == "Discrete":
                    _, old_log_prob_rollout = self.pi_lo[option].sample_log(states_op, actions_op)
                    old_log_prob_rollout = old_log_prob_rollout.detach()
                elif self.action_space == "Continuous": 
                    old_log_prob_rollout = self.pi_lo[option].sample_log(states_op, actions_op)
                    old_log_prob_rollout = old_log_prob_rollout.detach()   
                    
                self.value_function.train()
                self.pi_lo[option].train()
                
                num_steps_per_option = len(states_op)
                max_steps = self.num_epochs_pi_lo * (num_steps_per_option // self.minibatch_size)
                
                for _ in range(max_steps):
                    
                    minibatch_indices = np.random.choice(range(num_steps_per_option), self.minibatch_size, False)
                    batch_states = states_op[minibatch_indices]
                    batch_options = option_vector[minibatch_indices]
                    batch_actions = actions_op[minibatch_indices]
                    batch_returns = returns_op[minibatch_indices]
                    batch_advantage = advantage_op[minibatch_indices]  
                    batch_gammas = gammas_op[minibatch_indices]  
                    
                    if self.action_space == "Discrete":
                        log_prob, log_prob_rollout = self.pi_lo[option].sample_log(batch_states, batch_actions)
    
                    elif self.action_space == "Continuous": 
                        log_prob_rollout = self.pi_lo[option].sample_log(batch_states, batch_actions)
                        
                    batch_old_log_pi = old_log_prob_rollout[minibatch_indices]
                    
                    r = (torch.exp(log_prob_rollout - batch_old_log_pi)).squeeze()
                    L_clip = torch.minimum(r*batch_advantage, torch.clip(r, 1-self.epsilon, 1+self.epsilon)*batch_advantage)
                    batch_options = F.one_hot(batch_options, num_classes=self.option_dim)
                    L_vf = (self.value_function(batch_states, batch_options).squeeze() - batch_returns)**2
                
                    if self.action_space == "Discrete":
                        if Entropy:
                            S = (-1)*torch.sum(torch.exp(log_prob)*log_prob, 1)
                        else:
                            S = torch.zeros_like(torch.sum(torch.exp(log_prob)*log_prob, 1))
                            
                    elif self.action_space == "Continuous": 
                        if Entropy:
                            S = self.pi_lo[option].Distb(batch_states).entropy()
                        else:
                            S = torch.zeros_like(self.pi_lo[option].Distb(batch_states).entropy())       
                            
                    self.value_function_optimizer.zero_grad()
                    self.pi_lo_optimizer[option].zero_grad()
                    loss = (-1) * (L_clip - self.c1 * L_vf + self.c2 * S).mean()
                    loss.backward()
                    self.value_function_optimizer.step()
                    self.pi_lo_optimizer[option].step()  
            
            if self.Total_iter % self.pi_b_freq == 0:
                
                terminations_op = rollout_terminations[index_op[option]]
                advantage_option_op = rollout_advantage_option[index_op[option]]
                index_termination_true_op = np.where(terminations_op==1)[0]
                
                states_op_termination_true = states_op[index_termination_true_op]
                if len(states_op_termination_true)>=self.min_batch_size:
                    terminations_op_termination_true = terminations_op[index_termination_true_op]
                    advantage_option_op_termination_true = advantage_option_op[index_termination_true_op]
                    
                    advantage_option_op_termination_true = ((advantage_option_op_termination_true-advantage_option_op_termination_true.mean())/advantage_option_op_termination_true.std()).reshape(-1,1)
                    
                    self.pi_b[option].eval()
                    old_log_prob_termination, old_log_prob_rollout_termination = self.pi_b[option].sample_log(states_op_termination_true, terminations_op_termination_true)
                    old_log_prob_termination = old_log_prob_termination.detach()
                    old_log_prob_rollout_termination = old_log_prob_rollout_termination.detach()
                    
                    self.pi_b[option].train()
                    
                    num_steps_op_termination_true = len(states_op_termination_true)
                    max_steps_op_termination_true = self.num_epochs_pi_b * (num_steps_op_termination_true // self.minibatch_size)
                    
                    for _ in range(max_steps_op_termination_true):
                        
                        minibatch_indices_op_termination_true = np.random.choice(range(num_steps_op_termination_true), self.minibatch_size, False)
                        
                        batch_states_op_termination_true = states_op_termination_true[minibatch_indices_op_termination_true]
                        batch_terminations_op_termination_true = terminations_op_termination_true[minibatch_indices_op_termination_true]
                        batch_advantage_option_op_termination_true = advantage_option_op_termination_true[minibatch_indices_op_termination_true]
                        
                        log_prob_termination, log_prob_rollout_termination = self.pi_b[option].sample_log(batch_states_op_termination_true, batch_terminations_op_termination_true)
                        batch_old_log_prob_termination = old_log_prob_rollout_termination[minibatch_indices_op_termination_true]
                        
                        r_termination = torch.exp(log_prob_rollout_termination - batch_old_log_prob_termination)
                        Final_batch_adv_termination = (-1)*(batch_advantage_option_op_termination_true + self.eta*torch.ones_like(batch_advantage_option_op_termination_true))
                        L_termination_clip = torch.minimum(r_termination*Final_batch_adv_termination, torch.clip(r_termination, 1-self.epsilon, 1+self.epsilon)*Final_batch_adv_termination)
                        
                        if Entropy:
                            S_termination = (-1)*torch.sum(torch.exp(log_prob_termination)*log_prob_termination, 1)
                        else:
                            S_termination = torch.zeros_like(torch.sum(torch.exp(log_prob_termination)*log_prob_termination, 1))
                            
                        self.pi_b_optimizer[option].zero_grad()
                        loss_termination = (-1) * (L_termination_clip + self.c2 * S_termination).mean()
                        loss_termination.backward()
                        self.pi_b_optimizer[option].step() 
                    
        if self.Total_iter % self.pi_hi_freq == 0:
            
            index_termination_true = np.where(rollout_terminations==1)[0]
            
            states_termination_true = rollout_states[index_termination_true]
            if len(states_termination_true)>=self.min_batch_size:
                options_termination_true = rollout_options[index_termination_true]
                advantage_option_termination_true = rollout_advantage_option[index_termination_true]
                
                advantage_option_termination_true = ((advantage_option_termination_true-advantage_option_termination_true.mean())/advantage_option_termination_true.std()).reshape(-1,1)
                
                self.pi_hi.eval()
                old_log_prob_pi_hi, old_log_prob_rollout_pi_hi = self.pi_hi.sample_log(states_termination_true, options_termination_true)
                old_log_prob_pi_hi = old_log_prob_pi_hi.detach()
                old_log_prob_rollout_pi_hi = old_log_prob_rollout_pi_hi.detach()
                
                self.pi_hi.train()
                
                num_steps_termination_true = len(states_termination_true)        
                max_steps_termination_true = self.num_epochs_pi_hi * (num_steps_termination_true // self.minibatch_size)
                
                for _ in range(max_steps_termination_true):
                    
                    minibatch_indices_termination_true = np.random.choice(range(num_steps_termination_true), self.minibatch_size, False)
                    
                    batch_states_termination_true = states_termination_true[minibatch_indices_termination_true]
                    batch_options_termination_true = options_termination_true[minibatch_indices_termination_true]
                    batch_advantage_option_termination_true = advantage_option_termination_true[minibatch_indices_termination_true]
                    
                    log_prob_pi_hi, log_prob_rollout_pi_hi = self.pi_hi.sample_log(batch_states_termination_true, batch_options_termination_true)
                    batch_old_log_prob_pi_hi = old_log_prob_rollout_pi_hi[minibatch_indices_termination_true]
                    
                    r_pi_hi = torch.exp(log_prob_rollout_pi_hi - batch_old_log_prob_pi_hi)
                    L_pi_hi_clip = torch.minimum(r_pi_hi*batch_advantage_option_termination_true, torch.clip(r_pi_hi, 1-self.epsilon, 1+self.epsilon)*batch_advantage_option_termination_true)
                    
                    if Entropy:
                        S_pi_hi = (-1)*torch.sum(torch.exp(log_prob_pi_hi)*log_prob_pi_hi, 1)
                    else:
                        S_pi_hi = torch.zeros_like(torch.sum(torch.exp(log_prob_pi_hi)*log_prob_pi_hi, 1))
                            
                    self.pi_hi_optimizer.zero_grad()
                    loss_pi_hi = (-1) * (L_pi_hi_clip + self.c2 * S_pi_hi).mean()
                    loss_pi_hi.backward()
                    self.pi_hi_optimizer.step() 
                                   
    def save_actor(self, filename):
            torch.save(self.pi_hi.state_dict(), filename + "_pi_hi")
            torch.save(self.pi_hi_optimizer.state_dict(), filename + "_pi_hi_optimizer")
            
            for option in range(self.option_dim):
                torch.save(self.pi_lo[option].state_dict(), filename + f"_pi_lo_option_{option}")
                torch.save(self.pi_lo_optimizer[option].state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
                torch.save(self.pi_b[option].state_dict(), filename + f"_pi_b_option_{option}")
                torch.save(self.pi_b_optimizer[option].state_dict(), filename + f"_pi_b_optimizer_option_{option}")  
    
    def load_actor(self, filename):
            self.pi_hi.load_state_dict(torch.load(filename + "_pi_hi"))
            self.pi_hi_optimizer.load_state_dict(torch.load(filename + "_pi_hi_optimizer"))
            
            for option in range(self.option_dim):
                self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
                self.pi_lo_optimizer[option].load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
                self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))
                self.pi_b_optimizer[option].load_state_dict(torch.load(filename + f"_pi_b_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer"))            
        
     
        
        

        
        
        
        
        
        

            
            
        
            
            
            

        