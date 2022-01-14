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

from Buffer import H_V_trace_Buffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class H_GePPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, option_dim, termination_dim, 
                 num_steps_per_rollout=5000, pi_b_freq=3, pi_hi_freq=3, l_rate_pi_lo=1e-4, l_rate_pi_hi=3e-4 , l_rate_pi_b=3e-4, l_rate_critic=3e-4, 
                 gae_gamma = 0.99, epsilon = 0.1, c1 = 1, c2 = 1e-2, minibatch_size=64, num_epochs_pi_lo=2, num_epochs_pi_b=5, num_epochs_pi_hi=5, 
                 eta = 1e-7, min_batch_size = 64, N_old_policies = 4, c_trunc = 1):
        
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
        
        self.N_old_policies = N_old_policies
        self.c_trunc = c_trunc
        self.buffer = H_V_trace_Buffer(option_dim, N_old_policies)
        
        self.Total_t = 0
        self.Total_iter = 0
        self.states = []
        self.actions = []
        self.options = []
        self.terminations = []
        self.returns = []
        self.gammas = []
        
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
        
    def Generate_and_store_rollout(self, env):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.options = []
        self.terminations = []
        self.returns =  []
        self.gammas = []
        
        self.buffer.clear()
        
        while step < self.num_steps_per_rollout: 
            episode_states = []
            episode_actions = []
            episode_options = []
            episode_terminations = []
            episode_rewards = []
            episode_gammas = []

            state, done = env.reset(), False
            t=0
            episode_reward = 0
            
            initial_option = 0
            initial_b = 1
            option = H_GePPO.select_option(self, state, initial_b, initial_option)

            while not done and step < self.num_steps_per_rollout:            
                action = H_GePPO.select_action(self, state, option)
            
                self.states.append(state)
                self.actions.append(action)
                self.options.append(option)
                episode_states.append(state)
                episode_actions.append(action)
                episode_options.append(option)
                episode_gammas.append(self.gae_gamma**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
                
                termination = H_GePPO.select_termination(self, state, option)
                next_option = H_GePPO.select_option(self, state, termination, option)
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
            
            if self.action_space == "Discrete":
                episode_actions = torch.LongTensor(np.array(episode_actions))
            elif self.action_space == "Continuous":
                episode_actions = torch.FloatTensor(np.array(episode_actions))
            
            episode_options = torch.LongTensor(np.array(episode_options))
            episode_terminations = torch.LongTensor(np.array(episode_terminations))
            episode_rewards = torch.FloatTensor(np.array(episode_rewards))
            episode_gammas = torch.FloatTensor(np.array(episode_gammas))       
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.gammas.append(episode_gammas)
            self.returns.append(episode_returns)
            
            log_pi_lo_old = []
            log_pi_b_old = []
            
            self.pi_hi.eval()
            _, log_pi_hi_old = self.pi_hi.sample_log(episode_states, episode_options)
            
            for option in range(self.option_dim):
                self.pi_lo[option].eval()
                self.pi_b[option].eval()
                
                if self.action_space == "Discrete":
                    _, old_log_prob_rollout = self.pi_lo[option].sample_log(episode_states, episode_actions)
                    episode_old_log_prob = old_log_prob_rollout.detach()
                elif self.action_space == "Continuous": 
                    old_log_prob_rollout = self.pi_lo[option].sample_log(episode_states, episode_actions)
                    episode_old_log_prob = old_log_prob_rollout.detach()  
                    
                log_pi_lo_old.append(episode_old_log_prob)
                
                _, episode_old_log_prob_termination = self.pi_b[option].sample_log(episode_states, episode_terminations)
                log_pi_b_old.append(episode_old_log_prob_termination.detach())
                
            self.buffer.add(episode_states, episode_actions, episode_options, episode_terminations, episode_rewards, episode_returns, episode_gammas, log_pi_lo_old, log_pi_b_old, log_pi_hi_old.detach(), t)
                
        self.buffer.update_counters()
        
    def Lamda_H_ADV_trace_pi_lo(self, episode_states, episode_actions, episode_options, episode_log_pi_old, K):
        
        episode_log_pi = []
        for option in range(self.option_dim):  
            if self.action_space == "Discrete":
                _, log_prob_rollout = self.pi_lo[option].sample_log(episode_states, episode_actions)
                log_prob_rollout = log_prob_rollout.detach()

            elif self.action_space == "Continuous": 
                log_prob_rollout = self.pi_lo[option].sample_log(episode_states, episode_actions)
                log_prob_rollout = log_prob_rollout.detach()
                
            episode_log_pi.append(log_prob_rollout)
                        
        log_pi_old = torch.cat(episode_log_pi_old, 1)
        log_pi = torch.cat(episode_log_pi, 1)
        on_policy_episode_log_pi_old = log_pi_old.gather(1, episode_options.reshape(-1,1))
        on_policy_episode_log_pi = log_pi.gather(1, episode_options.reshape(-1,1))
        
        r = (torch.exp(on_policy_episode_log_pi - on_policy_episode_log_pi_old)).squeeze()
        r_trunc = torch.min(self.c_trunc*torch.ones_like(r), r)
        try:
            episode_lambdas = torch.FloatTensor([(r_trunc[:j]).prod() for j in range(K)])
        except:
            episode_lambdas = r_trunc
        
        return episode_lambdas
        
    def H_ADV_trace(self):
        
        stored_policies = self.buffer.size
        
        for i in range(stored_policies):
            
            states = self.buffer.Buffer[i]['states']
            actions = self.buffer.Buffer[i]['actions']
            options = self.buffer.Buffer[i]['options']
            terminations = self.buffer.Buffer[i]['terminations']
            rewards = self.buffer.Buffer[i]['rewards']
            gammas = self.buffer.Buffer[i]['gammas']
            episode_length = self.buffer.Buffer[i]['episode_length']
            
            log_pi_hi_old = self.buffer.Buffer[i]['log_pi_hi_old']
            
            log_pi_lo_old = []
            log_pi_b_old = []
            
            for option in range(self.option_dim):
                log_pi_lo_old.append(self.buffer.Buffer[i][f'log_pi_lo_old_{option}'])
                log_pi_b_old.append(self.buffer.Buffer[i][f'log_pi_b_old_{option}'])
                
            self.buffer.clear_Adv(i)
            
            for l in range(len(episode_length)): 
                
                episode_states = states[l]
                episode_actions = actions[l]
                episode_options = options[l]
                episode_terminations = terminations[l]
                episode_rewards = rewards[l]
                episode_gammas = gammas[l]
                
                episode_log_pi_hi_old = log_pi_hi_old[l] 
                
                episode_log_pi_lo_old = []
                episode_log_pi_b_old = []
                
                for option in range(self.option_dim):
                    episode_log_pi_lo_old.append(log_pi_lo_old[option][l])
                    episode_log_pi_b_old.append(log_pi_b_old[option][l])
                
                K = episode_length[l]
          
                self.value_function.eval()
                self.pi_hi.eval()
                
                episode_lambdas_pi_lo = H_GePPO.Lamda_H_ADV_trace_pi_lo(self, episode_states, episode_actions, episode_options, episode_log_pi_lo_old, K)
                
                episode_option_vector = torch.ones_like(episode_options, dtype=int)
                episode_options_encoded = F.one_hot(episode_options, num_classes=self.option_dim)
                current_values = self.value_function(episode_states, episode_options_encoded).detach()
                next_values = torch.cat((self.value_function(episode_states, episode_options_encoded)[1:], torch.FloatTensor([[0.]]))).detach()
                episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas_pi_lo)[:K-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(K)])
                
                current_values_option = 0
                next_values_option = 0
                for option_i in range(self.option_dim):
                    current_values_option_i = self.value_function(episode_states, F.one_hot(option_i*episode_option_vector, num_classes=self.option_dim)).detach()
                    next_values_option_i = torch.cat((self.value_function(episode_states, F.one_hot(option_i*episode_option_vector, num_classes=self.option_dim))[1:], torch.FloatTensor([[0.]]))).detach()
                    current_values_option += self.pi_hi(episode_states)[:,option_i].reshape(-1,1)*current_values_option_i
                    next_values_option += torch.cat((self.pi_hi(episode_states)[1:,option_i].reshape(-1,1), torch.FloatTensor([[0.]])))*next_values_option_i
                        
                episode_deltas_option = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values_option - current_values_option
                
                _, log_pi_hi = self.pi_hi.sample_log(episode_states, episode_options)
                r_pi_hi = (torch.exp(log_pi_hi.detach() - episode_log_pi_hi_old)).squeeze()
                r_trunc_pi_hi = torch.min(self.c_trunc*torch.ones_like(r_pi_hi), r_pi_hi)
                try:
                    episode_lambdas_pi_hi = torch.FloatTensor([(r_trunc_pi_hi[:j]).prod() for j in range(K)]) 
                except:
                    episode_lambdas_pi_hi = r_trunc_pi_hi
                    
                episode_advantage_option = torch.FloatTensor([((episode_gammas*episode_lambdas_pi_hi)[:K-j].unsqueeze(-1)*episode_deltas_option[j:]).sum() for j in range(K)])
             
                self.buffer.add_Adv(i, episode_advantage.detach(), episode_advantage_option.detach())
    
    def train(self, Entropy = False):
        
        stored_policies = self.buffer.size
        
        current_pi_lo = [[] for option in range(self.option_dim)]
        current_pi_b = [[] for option in range(self.option_dim)]
        current_pi_hi = []
        
        tot_steps = [[] for option in range(self.option_dim)]
        tot_terminations_op = [[] for option in range(self.option_dim)]
        tot_terminations = []
        
        for k in range(stored_policies):
            rollout_states = torch.cat(self.buffer.Buffer[k]['states'])
            rollout_actions = torch.cat(self.buffer.Buffer[k]['actions'])
            rollout_options = torch.cat(self.buffer.Buffer[k]['options'])
            rollout_terminations = torch.cat(self.buffer.Buffer[k]['terminations'])
            
            for option in range(self.option_dim):
                tot_steps[option].append(len(np.where(option==rollout_options)[0]))
                index_op = np.where(option==rollout_options)[0]
                op_termination_true = np.where(rollout_terminations[index_op]==1)[0]
                tot_terminations_op[option].append(len(op_termination_true))
                
            tot_terminations.append(len(np.where(rollout_terminations==1)[0]))
            
            _, log_pi_hi_rollout = self.pi_hi.sample_log(rollout_states, rollout_options)
            current_pi_hi.append(log_pi_hi_rollout.detach())
            
            for option in range(self.option_dim):
                
                if self.action_space == "Discrete":
                    _, log_pi_lo_rollout = self.pi_lo[option].sample_log(rollout_states, rollout_actions)
                    log_pi_lo_rollout = log_pi_lo_rollout.detach()
                elif self.action_space == "Continuous": 
                    log_pi_lo_rollout = self.pi_lo[option].sample_log(rollout_states, rollout_actions)
                    log_pi_lo_rollout = log_pi_lo_rollout.detach()
                    
                current_pi_lo[option].append(log_pi_lo_rollout)
                
                _, log_pi_b_rollout = self.pi_b[option].sample_log(rollout_states, rollout_terminations)
                current_pi_b[option].append(log_pi_b_rollout.detach())
                        
        for option in range(self.option_dim):
            
            num_steps_per_option = np.max(tot_steps[option]) 
            max_steps_pi_lo = self.num_epochs_pi_lo*(num_steps_per_option // self.minibatch_size)*stored_policies
            
            for s in range(max_steps_pi_lo):
                policy_number = np.random.randint(0, stored_policies)
                
                rollout_options = torch.cat(self.buffer.Buffer[policy_number]['options'])
                rollout_states = torch.cat(self.buffer.Buffer[policy_number]['states'])
                rollout_actions = torch.cat(self.buffer.Buffer[policy_number]['actions'])
                rollout_terminations = torch.cat(self.buffer.Buffer[policy_number]['terminations'])
                rollout_returns = torch.cat(self.buffer.Buffer[policy_number]['returns'])
                rollout_advantage = torch.cat(self.buffer.Buffer[policy_number]['advantage'])
                rollout_advantage_option = torch.cat(self.buffer.Buffer[policy_number]['advantage_option'])  
                rollout_gammas = torch.cat(self.buffer.Buffer[policy_number]['gammas'])   
                
                old_log_pi_lo_rollout = torch.cat(self.buffer.Buffer[policy_number][f'log_pi_lo_old_{option}'])  
                current_log_pi_lo_rollout = current_pi_lo[option][policy_number]
                    
                index_op = np.where(option==rollout_options)[0]
                states_op = rollout_states[index_op]
                if len(states_op)>=self.min_batch_size:
                    option_vector = rollout_options[index_op]
                    actions_op = rollout_actions[index_op]
                    returns_op = rollout_returns[index_op]
                    advantage_op = rollout_advantage[index_op]
                    gammas_op = rollout_gammas[index_op]
                    
                    advantage_op = (advantage_op-advantage_op.mean())/(advantage_op.std()+1e-6)
                    
                    old_log_pi_lo_op = old_log_pi_lo_rollout[index_op]
                    current_log_pi_lo_op = current_log_pi_lo_rollout[index_op]
                    
                    self.value_function.train()
                    self.pi_lo[option].train()
                    
                    num_steps = len(states_op)
                    minibatch_indices = np.random.choice(range(num_steps), self.minibatch_size, False)
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
  
                    batch_old_log_pi_lo = old_log_pi_lo_op[minibatch_indices]
                    batch_current_log_pi_lo = current_log_pi_lo_op[minibatch_indices]
                    
                    r = (torch.exp(log_prob_rollout - batch_old_log_pi_lo)).squeeze()
                    r_bound = (torch.exp(batch_current_log_pi_lo - batch_old_log_pi_lo)).squeeze()
                    clipping = self.epsilon*torch.ones_like(r_bound)
                    clipped_r = torch.where(r > r_bound+clipping, r_bound+clipping, r)
                    clipped_r = torch.where(r < r_bound-clipping, r_bound-clipping, r)
                    L_clip = torch.minimum(r*batch_advantage, clipped_r*batch_advantage)
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
                num_steps_per_option_termination_true = np.max(tot_terminations_op[option]) 
                max_steps_pi_b = self.num_epochs_pi_b*(num_steps_per_option_termination_true // self.minibatch_size)*stored_policies
                
                for s in range(max_steps_pi_b):
                    policy_number = np.random.randint(0, stored_policies)
                
                    rollout_options = torch.cat(self.buffer.Buffer[policy_number]['options'])
                    rollout_states = torch.cat(self.buffer.Buffer[policy_number]['states'])
                    rollout_actions = torch.cat(self.buffer.Buffer[policy_number]['actions'])
                    rollout_terminations = torch.cat(self.buffer.Buffer[policy_number]['terminations'])
                    rollout_returns = torch.cat(self.buffer.Buffer[policy_number]['returns'])
                    rollout_advantage = torch.cat(self.buffer.Buffer[policy_number]['advantage'])
                    rollout_advantage_option = torch.cat(self.buffer.Buffer[policy_number]['advantage_option'])  
                    rollout_gammas = torch.cat(self.buffer.Buffer[policy_number]['gammas'])   
                    
                    old_log_pi_b_rollout = torch.cat(self.buffer.Buffer[policy_number][f'log_pi_b_old_{option}'])  
                    current_log_pi_b_rollout = current_pi_b[option][policy_number]
                        
                    index_op = np.where(option==rollout_options)[0]
                    states_op = rollout_states[index_op]
                    
                    terminations_op = rollout_terminations[index_op]
                    advantage_option_op = rollout_advantage_option[index_op]
                    index_termination_true_op = np.where(terminations_op==1)[0]
                    
                    states_op_termination_true = states_op[index_termination_true_op]
                    
                    if len(states_op_termination_true)>=self.min_batch_size:
                        terminations_op_termination_true = terminations_op[index_termination_true_op]
                        advantage_option_op_termination_true = advantage_option_op[index_termination_true_op]       
                        
                        advantage_option_op_termination_true = ((advantage_option_op_termination_true-advantage_option_op_termination_true.mean())/advantage_option_op_termination_true.std()+1e-6).reshape(-1,1)
                        
                        old_log_pi_b_op_termination_true = old_log_pi_b_rollout[index_op]
                        current_log_pi_b_op_termination_true = current_log_pi_b_rollout[index_op]
                        
                        self.pi_b[option].train()
                        
                        num_steps_op_termination_true = len(states_op_termination_true)
                        minibatch_indices_op_termination_true = np.random.choice(range(num_steps_op_termination_true), self.minibatch_size, False)
                
                        batch_states_op_termination_true = states_op_termination_true[minibatch_indices_op_termination_true]
                        batch_terminations_op_termination_true = terminations_op_termination_true[minibatch_indices_op_termination_true]
                        batch_advantage_option_op_termination_true = advantage_option_op_termination_true[minibatch_indices_op_termination_true]
                        
                        log_prob_termination, log_prob_rollout_termination = self.pi_b[option].sample_log(batch_states_op_termination_true, batch_terminations_op_termination_true)
                        batch_old_log_prob_termination = old_log_pi_b_op_termination_true[minibatch_indices_op_termination_true]
                        batch_current_log_prob_termination = current_log_pi_b_op_termination_true[minibatch_indices_op_termination_true]
                        
                        r_termination = torch.exp(log_prob_rollout_termination - batch_old_log_prob_termination)
                        r_bound_termination = (torch.exp(batch_current_log_prob_termination - batch_old_log_prob_termination))
                        clipping_termination = self.epsilon*torch.ones_like(r_bound_termination)
                        clipped_r_termination = torch.where(r_termination > r_bound_termination+clipping_termination, r_bound_termination+clipping_termination, r_termination)
                        clipped_r_termination = torch.where(r_termination < r_bound_termination-clipping_termination, r_bound_termination-clipping_termination, r_termination)
                        Final_batch_adv_termination = (-1)*(batch_advantage_option_op_termination_true + self.eta*torch.ones_like(batch_advantage_option_op_termination_true))
                        L_termination_clip = torch.minimum(r_termination*Final_batch_adv_termination, clipped_r_termination*Final_batch_adv_termination)
                        
                        if Entropy:
                            S_termination = (-1)*torch.sum(torch.exp(log_prob_termination)*log_prob_termination, 1)
                        else:
                            S_termination = torch.zeros_like(torch.sum(torch.exp(log_prob_termination)*log_prob_termination, 1))
                            
                        self.pi_b_optimizer[option].zero_grad()
                        loss_termination = (-1) * (L_termination_clip + self.c2 * S_termination).mean()
                        loss_termination.backward()
                        self.pi_b_optimizer[option].step() 
                        
        if self.Total_iter % self.pi_hi_freq == 0:
            num_steps_termination_true = np.max(tot_terminations) 
            max_steps_pi_hi = self.num_epochs_pi_hi*(num_steps_termination_true // self.minibatch_size)*stored_policies   

            for s in range(max_steps_pi_hi):
                policy_number = np.random.randint(0, stored_policies)
                
                rollout_options = torch.cat(self.buffer.Buffer[policy_number]['options'])
                rollout_states = torch.cat(self.buffer.Buffer[policy_number]['states'])
                rollout_actions = torch.cat(self.buffer.Buffer[policy_number]['actions'])
                rollout_terminations = torch.cat(self.buffer.Buffer[policy_number]['terminations'])
                rollout_returns = torch.cat(self.buffer.Buffer[policy_number]['returns'])
                rollout_advantage = torch.cat(self.buffer.Buffer[policy_number]['advantage'])
                rollout_advantage_option = torch.cat(self.buffer.Buffer[policy_number]['advantage_option'])  
                rollout_gammas = torch.cat(self.buffer.Buffer[policy_number]['gammas'])   
                
                old_log_pi_hi_rollout = torch.cat(self.buffer.Buffer[policy_number]['log_pi_hi_old'])  
                current_log_pi_hi_rollout = current_pi_hi[policy_number]
                
                index_termination_true = np.where(rollout_terminations==1)[0]
                states_termination_true = rollout_states[index_termination_true]                 
                    
                if len(states_termination_true)>=self.min_batch_size:
                    options_termination_true = rollout_options[index_termination_true]
                    advantage_option_termination_true = rollout_advantage_option[index_termination_true]
                    
                    advantage_option_termination_true = ((advantage_option_termination_true-advantage_option_termination_true.mean())/advantage_option_termination_true.std()+1e-6).reshape(-1,1)
                    
                    old_log_pi_hi_termination_true = old_log_pi_hi_rollout[index_termination_true]
                    current_log_pi_hi_termination_true = current_log_pi_hi_rollout[index_termination_true]                    
                                    
                    self.pi_hi.train()
                    
                    num_steps_termination_true = len(states_termination_true)        
                
                    minibatch_indices_termination_true = np.random.choice(range(num_steps_termination_true), self.minibatch_size, False)
                    
                    batch_states_termination_true = states_termination_true[minibatch_indices_termination_true]
                    batch_options_termination_true = options_termination_true[minibatch_indices_termination_true]
                    batch_advantage_option_termination_true = advantage_option_termination_true[minibatch_indices_termination_true]
                    
                    log_prob_pi_hi, log_prob_rollout_pi_hi = self.pi_hi.sample_log(batch_states_termination_true, batch_options_termination_true)
                    batch_old_log_prob_pi_hi =  old_log_pi_hi_termination_true[minibatch_indices_termination_true]
                    batch_current_log_prob_pi_hi = current_log_pi_hi_termination_true[minibatch_indices_termination_true]
                    
                    r_pi_hi = torch.exp(log_prob_rollout_pi_hi - batch_old_log_prob_pi_hi)
                    r_bound_pi_hi = (torch.exp(batch_current_log_prob_pi_hi - batch_old_log_prob_pi_hi))
                    clipping_pi_hi = self.epsilon*torch.ones_like(r_bound_pi_hi)
                    clipped_r_pi_hi = torch.where(r_pi_hi > r_bound_pi_hi+clipping_pi_hi, r_bound_pi_hi+clipping_pi_hi, r_pi_hi)
                    clipped_r_pi_hi = torch.where(r_pi_hi < r_bound_pi_hi-clipping_pi_hi, r_bound_pi_hi-clipping_pi_hi, r_pi_hi)
                    L_pi_hi_clip = torch.minimum(r_pi_hi*batch_advantage_option_termination_true, clipped_r_pi_hi*batch_advantage_option_termination_true)
                    
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
        
     
        
        

        
        
        
        
        
        

            
            
        
            
            
            

        