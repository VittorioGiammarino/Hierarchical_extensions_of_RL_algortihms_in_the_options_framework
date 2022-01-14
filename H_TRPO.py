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

class H_TRPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, option_dim, termination_dim,
                 num_steps_per_rollout=5000, pi_b_freq=3, pi_hi_freq=3, l_rate_pi_lo=3e-4, l_rate_pi_hi=3e-4 , l_rate_pi_b=3e-4, l_rate_critic=3e-4, 
                 gae_gamma = 0.99, gae_lambda = 0.99, epsilon = 0.01, conj_grad_damping=0.1, lambda_ = 1e-3, eta = 1e-7, min_batch_size = 32):
        
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
        self.conj_grad_damping = conj_grad_damping
        self.lambda_ = lambda_
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
            distb = self.pi_lo[option].Distb(state)
            action = distb.sample().detach().cpu().numpy().flatten()
            return action
        
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
            option = H_TRPO.select_option(self, state, initial_b, initial_option)

            while not done and step < self.num_steps_per_rollout:            
                action = H_TRPO.select_action(self, state, option)
            
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
                
                termination = H_TRPO.select_termination(self, state, option)
                next_option = H_TRPO.select_option(self, state, termination, option)
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
            episode_actions = torch.LongTensor(np.array(episode_actions))
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
        rollout_actions = torch.LongTensor(np.array(self.actions))
        rollout_options = torch.LongTensor(np.array(self.options))
        rollout_terminations = torch.LongTensor(np.array(self.terminations))

        return rollout_states, rollout_actions, rollout_options, rollout_terminations
    
    def get_flat_grads(f, net):
        flat_grads = torch.cat([grad.view(-1) for grad in torch.autograd.grad(f, net.parameters(), create_graph=True)])
        return flat_grads
    
    def get_flat_params(net):
        return torch.cat([param.view(-1) for param in net.parameters()])
    
    def set_params(net, new_flat_params):
        start_idx = 0
        for param in net.parameters():
            end_idx = start_idx + np.prod(list(param.shape))
            param.data = torch.reshape(new_flat_params[start_idx:end_idx], param.shape)
            start_idx = end_idx
      
    def conjugate_gradient(Av_func, b, max_iter=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b - Av_func(x)
        p = r
        rsold = r.norm() ** 2
    
        for _ in range(max_iter):
            Ap = Av_func(p)
            alpha = rsold / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = r.norm() ** 2
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew   
        return x
    
    def rescale_and_linesearch(self, g, s, Hs, L, kld, old_params, option, max_iter=10, success_ratio=0.1):
        H_TRPO.set_params(self.pi_lo[option], old_params)
        L_old = L().detach()
        max_kl = self.epsilon
        
        eta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))
    
        for _ in range(max_iter):
            new_params = old_params + eta * s
    
            H_TRPO.set_params(self.pi_lo[option], new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta * s)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params, False
    
            eta *= 0.7
    
        print(f"The line search for pi_lo {option} was failed!")
        return old_params, True
    
    def rescale_and_linesearch_termination(self, g, s, Hs, L, kld, old_params, option, max_iter=10, success_ratio=0.1):
        H_TRPO.set_params(self.pi_b[option], old_params)
        L_old = L().detach()
        max_kl = self.epsilon
        
        eta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))
    
        for _ in range(max_iter):
            new_params = old_params + eta * s
    
            H_TRPO.set_params(self.pi_b[option], new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta * s)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params, False
    
            eta *= 0.7
    
        print(f"The line search for pi_b option {option} was failed!")
        return old_params, True
    
    def rescale_and_linesearch_pi_hi(self, g, s, Hs, L, kld, old_params, max_iter=10, success_ratio=0.1):
        H_TRPO.set_params(self.pi_hi, old_params)
        L_old = L().detach()
        max_kl = self.epsilon
        
        eta = torch.sqrt((2 * max_kl) / torch.dot(s, Hs))
    
        for _ in range(max_iter):
            new_params = old_params + eta * s
    
            H_TRPO.set_params(self.pi_hi, new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta * s)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl:
                return new_params, False
    
            eta *= 0.7
    
        print("The line search for pi_hi was failed!")
        return old_params, True
    
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
                
                advantage_op = (advantage_op-advantage_op.mean())/advantage_op.std()      
                
                self.value_function.train()
                old_params = H_TRPO.get_flat_params(self.value_function).detach()
                option_vector = F.one_hot(option_vector, num_classes=self.option_dim)
                old_v = self.value_function(states_op, option_vector).detach()
                
                def constraint():
                    return ((old_v - self.value_function(states_op, option_vector))**2).mean()
                
                gradient_constraint = H_TRPO.get_flat_grads(constraint(), self.value_function)
                
                def Hv(v):
                    hessian_v = H_TRPO.get_flat_grads(torch.dot(gradient_constraint, v), self.value_function).detach()
                    return hessian_v
                
                gradient = H_TRPO.get_flat_grads(((-1)*(self.value_function(states_op, option_vector).squeeze() - returns_op)**2).mean(), self.value_function).detach()
                s = H_TRPO.conjugate_gradient(Hv, gradient).detach()
                Hessian_s = Hv(s).detach()
                alpha = torch.sqrt(2*self.epsilon/torch.dot(s,Hessian_s))
                new_params = old_params + alpha*s
                H_TRPO.set_params(self.value_function, new_params)            
                
                self.pi_lo[option].train()
                old_params = H_TRPO.get_flat_params(self.pi_lo[option]).detach()
                
                if self.action_space == "Discrete":
                    old_log_prob, old_log_prob_rollout = self.pi_lo[option].sample_log(states_op, actions_op)
                elif self.action_space == "Continuous": 
                    old_distb = self.pi_lo[option].Distb(states_op)
            
                def L():
                    if self.action_space == "Discrete":
                        _, log_prob_rollout = self.pi_lo[option].sample_log(states_op, actions_op)
                        return (advantage_op*torch.exp((log_prob_rollout - old_log_prob_rollout.detach()).squeeze())).mean()
                    elif self.action_space == "Continuous": 
                        distb = self.pi_lo[option].Distb(states_op)      
                        return (advantage_op*torch.exp(distb.log_prob(actions_op) - old_distb.log_prob(actions_op).detach())).mean()
                
                def kld():    
                    if self.action_space == "Discrete":
                        prob = self.pi_lo[option](states_op)
                        divKL = F.kl_div(old_log_prob.detach(), prob, reduction = 'batchmean')
                        return divKL
                    
                    elif self.action_space == "Continuous":  
                        distb = self.pi_lo[option].Distb(states_op)  
                        old_mean = old_distb.mean.detach()
                        old_cov = old_distb.covariance_matrix.sum(-1).detach()
                        mean = distb.mean
                        cov = distb.covariance_matrix.sum(-1)
                        return (0.5)*((old_cov/cov).sum(-1)+(((old_mean - mean) ** 2)/cov).sum(-1)-self.action_dim + torch.log(cov).sum(-1) - torch.log(old_cov).sum(-1)).mean()
                    
                grad_kld_old_param = H_TRPO.get_flat_grads(kld(), self.pi_lo[option])
            
                def Hv(v):
                    hessian_v = H_TRPO.get_flat_grads(torch.dot(grad_kld_old_param, v), self.pi_lo[option]).detach()
                    return hessian_v + self.conj_grad_damping*v
            
                gradient = H_TRPO.get_flat_grads(L(), self.pi_lo[option]).detach()
                s = H_TRPO.conjugate_gradient(Hv, gradient).detach()
                Hs = Hv(s).detach()
                new_params, Failed = H_TRPO.rescale_and_linesearch(self, gradient, s, Hs, L, kld, old_params, option)
            
                if Entropy and not Failed:
                    if self.action_space == "Discrete":
                        _, entropy_log_prob = self.pi_lo[option].sample_log(states_op, actions_op)
                        discounted_casual_entropy = ((-1)*gammas_op*entropy_log_prob).mean()
                        gradient_discounted_casual_entropy = H_TRPO.get_flat_grads(discounted_casual_entropy, self.pi_lo[option])
                        new_params += self.lambda_*gradient_discounted_casual_entropy
                    elif self.action_space == "Continuous":  
                        discounted_casual_entropy = ((-1)*gammas_op*self.pi_lo[option].Distb(states_op).log_prob(actions_op)).mean()
                        gradient_discounted_casual_entropy = H_TRPO.get_flat_grads(discounted_casual_entropy, self.pi_lo[option])
                        new_params += self.lambda_*gradient_discounted_casual_entropy
                    
                H_TRPO.set_params(self.pi_lo[option], new_params)
            
            if self.Total_iter % self.pi_b_freq == 0:
                
                terminations_op = rollout_terminations[index_op[option]]
                advantage_option_op = rollout_advantage_option[index_op[option]]
                index_termination_true_op = np.where(terminations_op==1)[0]
                
                states_op_termination_true = states_op[index_termination_true_op]
                if len(states_op_termination_true)>=self.min_batch_size:
                    terminations_op_termination_true = terminations_op[index_termination_true_op]
                    advantage_option_op_termination_true = advantage_option_op[index_termination_true_op]
                    gammas_op_termination_true = gammas_op[index_termination_true_op]
                    
                    advantage_option_op_termination_true = ((advantage_option_op_termination_true-advantage_option_op_termination_true.mean())/advantage_option_op_termination_true.std()).reshape(-1,1)
                    
                    self.pi_b[option].train()
                    old_params_termination = H_TRPO.get_flat_params(self.pi_b[option]).detach() 
                    old_log_prob_termination, old_log_prob_rollout_termination = self.pi_b[option].sample_log(states_op_termination_true, terminations_op_termination_true)
                    
                    def L_termination():
                        log_prob_termination, log_prob_rollout_termination = self.pi_b[option].sample_log(states_op_termination_true, terminations_op_termination_true)
                        r_termination = torch.exp(log_prob_rollout_termination - old_log_prob_rollout_termination.detach())
                        Final_adv_termination = (-1)*(advantage_option_op_termination_true + self.eta*torch.ones_like(advantage_option_op_termination_true))                     
                        return (Final_adv_termination*r_termination).mean()
                    
                    def kld_termination():    
                        prob_termination = self.pi_b[option](states_op_termination_true)
                        divKL = F.kl_div(old_log_prob_termination.detach(), prob_termination, reduction = 'batchmean')
                        return divKL
                    
                    grad_kld_termination_old_param = H_TRPO.get_flat_grads(kld_termination(), self.pi_b[option])
                
                    def Hv(v):
                        hessian_v = H_TRPO.get_flat_grads(torch.dot(grad_kld_termination_old_param, v), self.pi_b[option]).detach()
                        return hessian_v + self.conj_grad_damping*v
                
                    gradient_termination = H_TRPO.get_flat_grads(L_termination(), self.pi_b[option]).detach()
                    s = H_TRPO.conjugate_gradient(Hv, gradient_termination).detach()
                    Hs = Hv(s).detach()
                    new_params_termination, Failed = H_TRPO.rescale_and_linesearch_termination(self, gradient_termination, s, Hs, L_termination, kld_termination, old_params_termination, option)
                
                    if Entropy and not Failed:
                        _, entropy_log_prob = self.pi_b[option].sample_log(states_op_termination_true, terminations_op_termination_true)
                        discounted_causal_entropy_termination = ((-1)*gammas_op_termination_true*entropy_log_prob).mean()
                        gradient_discounted_causal_entropy_termination = H_TRPO.get_flat_grads(discounted_causal_entropy_termination, self.pi_b[option])
                        new_params_termination += self.lambda_*gradient_discounted_causal_entropy_termination
    
                    H_TRPO.set_params(self.pi_b[option], new_params_termination)
                
        if self.Total_iter % self.pi_hi_freq == 0:
            
            index_termination_true = np.where(rollout_terminations==1)[0]
            states_termination_true = rollout_states[index_termination_true]
            if len(states_termination_true)>=self.min_batch_size:
                options_termination_true = rollout_options[index_termination_true]
                advantage_option_termination_true = rollout_advantage_option[index_termination_true]
                gammas_termination_true = rollout_gammas[index_termination_true]
                
                advantage_option_termination_true = ((advantage_option_termination_true-advantage_option_termination_true.mean())/advantage_option_termination_true.std()).reshape(-1,1)
                
                self.pi_hi.train()
                old_params_pi_hi = H_TRPO.get_flat_params(self.pi_hi).detach() 
                old_log_prob_pi_hi, old_log_prob_rollout_pi_hi = self.pi_hi.sample_log(states_termination_true, options_termination_true)
                
                def L_pi_hi():
                    log_prob_pi_hi, log_prob_rollout_pi_hi = self.pi_hi.sample_log(states_termination_true, options_termination_true)
                    r_pi_hi = torch.exp(log_prob_rollout_pi_hi - old_log_prob_rollout_pi_hi.detach())                    
                    return (advantage_option_termination_true*r_pi_hi).mean()
                
                def kld_pi_hi():    
                    prob_pi_hi = self.pi_hi(states_termination_true)
                    divKL = F.kl_div(old_log_prob_pi_hi.detach(), prob_pi_hi, reduction = 'batchmean')
                    return divKL
                
                grad_kld_pi_hi_old_param = H_TRPO.get_flat_grads(kld_pi_hi(), self.pi_hi)
            
                def Hv(v):
                    hessian_v = H_TRPO.get_flat_grads(torch.dot(grad_kld_pi_hi_old_param, v), self.pi_hi).detach()
                    return hessian_v + self.conj_grad_damping*v
            
                gradient_pi_hi = H_TRPO.get_flat_grads(L_pi_hi(), self.pi_hi).detach()
                s = H_TRPO.conjugate_gradient(Hv, gradient_pi_hi).detach()
                Hs = Hv(s).detach()
                new_params_pi_hi, Failed = H_TRPO.rescale_and_linesearch_pi_hi(self, gradient_pi_hi, s, Hs, L_pi_hi, kld_pi_hi, old_params_pi_hi)
            
                if Entropy and not Failed:
                    _, entropy_log_prob = self.pi_hi.sample_log(states_termination_true, options_termination_true)
                    discounted_causal_entropy_pi_hi = ((-1)*gammas_termination_true*entropy_log_prob).mean()
                    gradient_discounted_causal_entropy_pi_hi = H_TRPO.get_flat_grads(discounted_causal_entropy_pi_hi, self.pi_hi)
                    new_params_pi_hi += self.lambda_*gradient_discounted_causal_entropy_pi_hi
    
                H_TRPO.set_params(self.pi_hi, new_params_pi_hi)        
            

    def save_actor(self, filename):
            torch.save(self.pi_hi.state_dict(), filename + "_pi_hi")
            
            for option in range(self.option_dim):
                torch.save(self.pi_lo[option].state_dict(), filename + f"_pi_lo_option_{option}")
                torch.save(self.pi_b[option].state_dict(), filename + f"_pi_b_option_{option}")
    
    def load_actor(self, filename):
            self.pi_hi.load_state_dict(torch.load(filename + "_pi_hi"))
            
            for option in range(self.option_dim):
                self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
                self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))       
        
        
        
        
        
        
        
        
        

            
            
        
            
            
            

        