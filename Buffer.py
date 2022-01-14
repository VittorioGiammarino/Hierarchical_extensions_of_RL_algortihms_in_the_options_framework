#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 09:29:37 2021

@author: vittorio
"""
import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cost = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def add(self, state, action, next_state, reward, cost, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.cost[self.ptr] = cost
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
            )

class V_trace_Buffer(object):
    def __init__(self, max_size=4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.Buffer = [{} for i in range(max_size)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.max_size):
            self.Buffer[i]['states'] = []
            self.Buffer[i]['actions'] = []
            self.Buffer[i]['rewards'] = []
            self.Buffer[i]['returns'] = []
            self.Buffer[i]['gammas'] = []
            self.Buffer[i]['log_pi_old'] = []
            self.Buffer[i]['episode_length'] = []
            self.Buffer[i]['advantage'] = []
            
    def clear(self):
        self.Buffer[self.ptr]['states'] = []
        self.Buffer[self.ptr]['actions'] = []
        self.Buffer[self.ptr]['rewards'] = []
        self.Buffer[self.ptr]['returns'] = []
        self.Buffer[self.ptr]['gammas'] = []
        self.Buffer[self.ptr]['log_pi_old'] = []
        self.Buffer[self.ptr]['episode_length'] = []
                
    def add(self, states, actions, rewards, returns, gammas, log_pi_old, episode_length):
        self.Buffer[self.ptr]['states'].append(states)
        self.Buffer[self.ptr]['actions'].append(actions)
        self.Buffer[self.ptr]['rewards'].append(rewards)
        self.Buffer[self.ptr]['returns'].append(returns)
        self.Buffer[self.ptr]['gammas'].append(gammas)
        self.Buffer[self.ptr]['log_pi_old'].append(log_pi_old)
        self.Buffer[self.ptr]['episode_length'].append(episode_length)
        
    def clear_Adv(self, i):
        self.Buffer[i]['advantage'] = []
        
    def add_Adv(self, i, advantage):
        self.Buffer[i]['advantage'].append(advantage)
        
    def update_counters(self):
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

class H_V_trace_Buffer(object):
    def __init__(self, num_options, max_size=4):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.num_options = num_options
        
        self.Buffer = [{} for i in range(max_size)]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(self.max_size):
            self.Buffer[i]['states'] = []
            self.Buffer[i]['actions'] = []
            self.Buffer[i]['options'] = []
            self.Buffer[i]['terminations'] = []
            self.Buffer[i]['rewards'] = []
            self.Buffer[i]['returns'] = []
            self.Buffer[i]['gammas'] = []
            self.Buffer[i]['episode_length'] = []
            self.Buffer[i]['advantage'] = []
            self.Buffer[i]['advantage_option'] = []
            
            self.Buffer[i]['log_pi_hi_old'] = []
            for option in range(num_options):
                self.Buffer[i][f'log_pi_lo_old_{option}'] = []
                self.Buffer[i][f'log_pi_b_old_{option}'] = []
            
    def clear(self):
        self.Buffer[self.ptr]['states'] = []
        self.Buffer[self.ptr]['actions'] = []
        self.Buffer[self.ptr]['options'] = []
        self.Buffer[self.ptr]['terminations'] = []
        self.Buffer[self.ptr]['rewards'] = []
        self.Buffer[self.ptr]['returns'] = []
        self.Buffer[self.ptr]['gammas'] = []
        self.Buffer[self.ptr]['episode_length'] = []
        
        self.Buffer[self.ptr]['log_pi_hi_old'] = []
        for option in range(self.num_options):
            self.Buffer[self.ptr][f'log_pi_lo_old_{option}'] = []
            self.Buffer[self.ptr][f'log_pi_b_old_{option}'] = []
                
    def add(self, states, actions, options, terminations, rewards, returns, gammas, log_pi_lo_old, log_pi_b_old, log_pi_hi_old, episode_length):
        self.Buffer[self.ptr]['states'].append(states)
        self.Buffer[self.ptr]['actions'].append(actions)
        self.Buffer[self.ptr]['options'].append(options)
        self.Buffer[self.ptr]['terminations'].append(terminations)
        self.Buffer[self.ptr]['rewards'].append(rewards)
        self.Buffer[self.ptr]['returns'].append(returns)
        self.Buffer[self.ptr]['gammas'].append(gammas)
        self.Buffer[self.ptr]['episode_length'].append(episode_length)
        
        self.Buffer[self.ptr]['log_pi_hi_old'].append(log_pi_hi_old)
        for option in range(self.num_options):
            self.Buffer[self.ptr][f'log_pi_lo_old_{option}'].append(log_pi_lo_old[option])
            self.Buffer[self.ptr][f'log_pi_b_old_{option}'].append(log_pi_b_old[option])       
        
    def clear_Adv(self, i):
        self.Buffer[i]['advantage'] = []
        self.Buffer[i]['advantage_option'] = []
        
    def add_Adv(self, i, advantage, advantage_option):
        self.Buffer[i]['advantage'].append(advantage)
        self.Buffer[i]['advantage_option'].append(advantage_option)
        
    def update_counters(self):
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)