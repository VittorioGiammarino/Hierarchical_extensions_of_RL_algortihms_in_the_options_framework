#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:55:49 2021

@author: vittorio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TanhGaussianHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(TanhGaussianHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim),
            )
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.high = max_action
            self.low = -max_action
        		
        def forward(self, state):        
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std) 
            return mean.cpu(), std.cpu()
        
        def squash(self, raw_values):
            squashed = ((torch.tanh(raw_values.detach())+1)/2.0)*(self.high-self.low)+self.low
            for a in range(self.action_dim):
                squashed[:,a] = torch.clamp(squashed[:,a], self.low[a], self.high[a])
            return squashed.float()
        
        def unsquash(self, values):
            normed_values = (values - self.low)/(self.high - self.low)*2.0 - 1.0
            stable_normed_values = torch.clamp(normed_values, -1+1e-4, 1-1e-4)
            unsquashed = torch.atanh(stable_normed_values)
            return unsquashed.float()
        
        def sample(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            y = torch.tanh(x)        
            action = self.squash(x)
            log_prob = torch.clamp(normal.log_prob(x), -5, 5)
            log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = self.squash(mean)
            return action, log_prob, mean 
        
        def sample_log(self, state, action):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = self.unsquash(action)
            y = torch.tanh(x) 
            log_prob = torch.clamp(normal.log_prob(x), -5, 5)
            log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
            return log_prob
        
        def sample_SAC_continuous(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            y = torch.tanh(x)        
            action = y*self.high[0]
            log_prob = normal.log_prob(x)
            log_prob -= torch.log(self.high[0]*(1 - y.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean)*self.high[0]
            return action, log_prob, mean 
        
        def Distb(self, state):
            mean = self.net(state)
            log_std = self.log_std.clamp(-20,2)
            std = torch.exp(log_std)
            cov_mtx = torch.eye(self.action_dim) * (std ** 2)
            distb = torch.distributions.MultivariateNormal(mean, cov_mtx)

            return distb
        
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(TanhGaussianHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(torch.clamp(self.l3(b),-10,10))            
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10)) 
            
            prob = self.forward(state)
            m = Categorical(prob)
            termination = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return termination, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, termination):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10)) 
                        
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return log_prob, log_prob_sampled.reshape(-1,1)            
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(TanhGaussianHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(torch.clamp(self.l3(o),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10)) 
            
            prob = self.forward(state)
            m = Categorical(prob)
            option = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return option, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, option):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10)) 
                        
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return log_prob, log_prob_sampled.reshape(-1,1)


class DeepDeterministicHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim, max_action):
            super(DeepDeterministicHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.action_dim = action_dim
            self.net = torch.nn.Sequential(
                torch.nn.Linear(state_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, action_dim),
            )
            
            self.action_dim = action_dim
            self.state_dim = state_dim
            self.max_action = max_action
            
            self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            
        def forward(self, states):
            mean = self.net(states)     
            return self.max_action[0] * torch.tanh(mean)
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(DeepDeterministicHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(torch.clamp(self.l3(b),-10,10))            
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10)) 
            
            prob = self.forward(state)
            m = Categorical(prob)
            termination = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return termination, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, termination):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10)) 
                        
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return log_prob, log_prob_sampled.reshape(-1,1)            
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(DeepDeterministicHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(self.l3(o))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            option = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return option, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, option):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return log_prob, log_prob_sampled.reshape(-1,1)


class SoftmaxHierarchicalActor:
    class NN_PI_LO(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(SoftmaxHierarchicalActor.NN_PI_LO, self).__init__()
            
            self.l1 = nn.Linear(state_dim, 128)
            nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(128,128)
            nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.l3 = nn.Linear(128,action_dim)
            nn.init.uniform_(self.l3.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            a = self.l1(state)
            a = F.relu(self.l2(a))
            return self.lS(torch.clamp(self.l3(a),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            action = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(action)),action]
            
            return action, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, action):
            self.log_Soft = nn.LogSoftmax(dim=1)
            a = self.l1(state)
            a = F.relu(self.l2(a))
            log_prob = self.log_Soft(torch.clamp(self.l3(a),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(action)), action]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
            
        
    class NN_PI_B(nn.Module):
        def __init__(self, state_dim, termination_dim):
            super(SoftmaxHierarchicalActor.NN_PI_B, self).__init__()
            
            self.l1 = nn.Linear(state_dim,10)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(10,10)
            self.l3 = nn.Linear(10,termination_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)
            
        def forward(self, state):
            b = self.l1(state)
            b = F.relu(self.l2(b))
            return self.lS(torch.clamp(self.l3(b),-10,10))            
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            termination = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return termination, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, termination):
            self.log_Soft = nn.LogSoftmax(dim=1)
            b = self.l1(state)
            b = F.relu(self.l2(b))
            log_prob = self.log_Soft(torch.clamp(self.l3(b),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(termination)), termination]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
    class NN_PI_HI(nn.Module):
        def __init__(self, state_dim, option_dim):
            super(SoftmaxHierarchicalActor.NN_PI_HI, self).__init__()
            
            self.l1 = nn.Linear(state_dim,5)
            # nn.init.uniform_(self.l1.weight, -0.5, 0.5)
            self.l2 = nn.Linear(5,5)
            self.l3 = nn.Linear(5,option_dim)
            # nn.init.uniform_(self.l2.weight, -0.5, 0.5)
            self.lS = nn.Softmax(dim=1)

        def forward(self, state):
            o = self.l1(state)
            o = F.relu(self.l2(o))
            return self.lS(torch.clamp(self.l3(o),-10,10))
        
        def sample(self, state):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
            
            prob = self.forward(state)
            m = Categorical(prob)
            option = m.sample()
            
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return option, log_prob_sampled.reshape(-1,1)
        
        def sample_log(self, state, option):
            self.log_Soft = nn.LogSoftmax(dim=1)
            o = self.l1(state)
            o = F.relu(self.l2(o))
            log_prob = self.log_Soft(torch.clamp(self.l3(o),-10,10))
                        
            log_prob_sampled = log_prob[torch.arange(len(option)), option]
            
            return log_prob, log_prob_sampled.reshape(-1,1)
        
class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, option_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim + option_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sao))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action, option):
		sao = torch.cat([state, action, option], 1)

		q1 = F.relu(self.l1(sao))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class Critic_discrete(nn.Module):
	def __init__(self, state_dim, action_cardinality, option_dim):
		super(Critic_discrete, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + option_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_cardinality)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + option_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, action_cardinality)


	def forward(self, state, option):
		so = torch.cat([state, option], 1)

		q1 = F.relu(self.l1(so))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(so))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, option):
		so = torch.cat([state, option], 1)

		q1 = F.relu(self.l1(so))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1
    
class Critic_flat(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic_flat, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class Critic_flat_discrete(nn.Module):
    def __init__(self, state_dim, action_cardinality):
        super(Critic_flat_discrete, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_cardinality)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, action_cardinality)


    def forward(self, state):
        
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(state))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state):
        
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
    
class Value_net(nn.Module):
    def __init__(self, state_dim):
        super(Value_net, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1
    
class Value_net_H(nn.Module):
    def __init__(self, state_dim, option_dim):
        super(Value_net_H, self).__init__()
        # Value_net architecture
        self.l1 = nn.Linear(state_dim + option_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, option):
        so = torch.cat([state, option], 1)
        
        q1 = F.relu(self.l1(so))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)    
        return q1
    
class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        # architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        return torch.sigmoid(self.get_logits(state, action))

    def get_logits(self, state, action):
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        sa = torch.cat([state, action], 1)
        d = F.relu(self.l1(sa))
        d = F.relu(self.l2(d))
        d = self.l3(d)
        return d