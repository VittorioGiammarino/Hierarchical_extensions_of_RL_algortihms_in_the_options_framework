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

from models import TanhGaussianHierarchicalActor
from models import SoftmaxHierarchicalActor
from models import Value_net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UATRPO:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, 
                 num_steps_per_rollout=5000,  gae_gamma = 0.99, gae_lambda = 0.99, epsilon = 0.03, conj_grad_damping=0.1, 
                 random_projections = 200, beta = 0.9, alpha = 0.05, c=6e-4, lambda_ = 1e-3):
        
        if np.isinf(action_space_cardinality):
            self.actor = TanhGaussianHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            self.action_space = "Continuous"
            
        else:
            self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            self.action_space = "Discrete"
            
        self.value_function = Value_net(state_dim).to(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.d = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
        self.random_projections = random_projections
        self.Random_matrix = torch.randn((self.d, self.random_projections), dtype=torch.float32)
        self.beta = beta
        self.beta_t = beta
        self.Y_F_old = torch.zeros_like(self.Random_matrix)
        self.Y_S_old = torch.zeros_like(self.Random_matrix)
        self.alpha = alpha
        self.trade_off = c
        
        self.num_steps_per_rollout = num_steps_per_rollout
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.conj_grad_damping = conj_grad_damping
        self.lambda_ = lambda_
        
        self.Total_t = 0
        self.Total_iter = 0
        self.Total_trajs = 0
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
        self.Total_trajs = 0
        
    def select_action(self, state):
        self.actor.eval()
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            action, _ = self.actor.sample(state)
            return int((action).cpu().data.numpy().flatten())
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            distb = self.actor.Distb(state)
            action = distb.sample().detach().cpu().numpy().flatten()
            return action
        
    def GAE(self, env, GAIL = False, Discriminator = None):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        while step < self.num_steps_per_rollout: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout:            
                action = UATRPO.select_action(self, state)
            
                self.states.append(state)
                self.actions.append(action)
                episode_states.append(state)
                episode_actions.append(action)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                self.Total_trajs += 1
                
            episode_states = torch.FloatTensor(np.array(episode_states))
            episode_actions = torch.FloatTensor(np.array(episode_actions))
            episode_rewards = torch.FloatTensor(np.array(episode_rewards))
            episode_gammas = torch.FloatTensor(np.array(episode_gammas))
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas))       
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)])
            episode_returns = episode_discounted_returns/episode_gammas
            
            self.returns.append(episode_returns)
            self.value_function.eval()
            current_values = self.value_function(episode_states).detach()
            next_values = torch.cat((self.value_function(episode_states)[1:], torch.FloatTensor([[0.]]))).detach()
            episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
            episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            self.advantage.append(episode_advantage)
            self.gammas.append(episode_gammas)
            
        rollout_states = torch.FloatTensor(np.array(self.states))
        rollout_actions = torch.FloatTensor(np.array(self.actions))

        return rollout_states, rollout_actions
    
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
    
    def rescale_and_linesearch(self, g, eta_v, Fv, Sv, L, Rn2, kld, old_params, max_iter=10, success_ratio=0.1):
        UATRPO.set_params(self.actor, old_params)
        L_old = L().detach()
        max_kl = self.epsilon
    
        for _ in range(max_iter):
                        
            # vFv_cur = 0.5 * torch.tensordot(eta_v, Fv, 1)
            # vSv_cur = 0.5 * torch.tensordot(eta_v, Sv, 1) 
            
            # delta_mult = (vFv_cur /(vFv_cur + self.trade_off * Rn2 * vSv_cur))
            delta_mult = 1
            
            new_params = old_params + eta_v
    
            UATRPO.set_params(self.actor, new_params)
            kld_new = kld().detach()
    
            L_new = L().detach()
    
            actual_improv = L_new - L_old
            approx_improv = torch.dot(g, eta_v)
            ratio = actual_improv / approx_improv
    
            if ratio > success_ratio and actual_improv > 0 and kld_new < max_kl*delta_mult:
                return new_params, False
    
            eta_v *= 0.7
    
        print("The line search was failed!")
        return old_params, True
    
    def train(self, Entropy=False):
        
        rollout_states = torch.FloatTensor(np.array(self.states))
        
        if self.action_space == "Discrete":
            rollout_actions = torch.LongTensor(np.array(self.actions))
        elif self.action_space == "Continuous":
            rollout_actions = torch.FloatTensor(np.array(self.actions))
            
        rollout_returns = torch.cat(self.returns)
        rollout_advantage = torch.cat(self.advantage)
        rollout_gammas = torch.cat(self.gammas)        
        
        rollout_advantage = (rollout_advantage-rollout_advantage.mean())/rollout_advantage.std()
        
        self.value_function.train()
        old_params = UATRPO.get_flat_params(self.value_function).detach()
        old_v = self.value_function(rollout_states).detach()
        
        def constraint():
            return ((old_v - self.value_function(rollout_states))**2).mean()
        
        gradient_constraint = UATRPO.get_flat_grads(constraint(), self.value_function)
        
        def Hv(v):
            hessian_v = UATRPO.get_flat_grads(torch.dot(gradient_constraint, v), self.value_function).detach()
            return hessian_v
        
        gradient = UATRPO.get_flat_grads(((-1)*(self.value_function(rollout_states).squeeze() - rollout_returns)**2).mean(), self.value_function).detach()
        s = UATRPO.conjugate_gradient(Hv, gradient).detach()
        Hessian_s = Hv(s).detach()
        alpha = torch.sqrt(2*self.epsilon/torch.dot(s,Hessian_s))
        new_params = old_params + alpha*s
        UATRPO.set_params(self.value_function, new_params)
        
        self.actor.train()
        old_params = UATRPO.get_flat_params(self.actor).detach()
        
        if self.action_space == "Discrete":
            old_log_prob, old_log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
        elif self.action_space == "Continuous": 
            old_distb = self.actor.Distb(rollout_states)
        
        def L():
            if self.action_space == "Discrete":
                _, log_prob_rollout = self.actor.sample_log(rollout_states, rollout_actions)
                return (rollout_advantage*torch.exp((log_prob_rollout - old_log_prob_rollout.detach()).squeeze())).mean()
            elif self.action_space == "Continuous": 
                distb = self.actor.Distb(rollout_states)
                return (rollout_advantage*torch.exp(distb.log_prob(rollout_actions) - old_distb.log_prob(rollout_actions).detach())).mean()
            
        def kld():
            if self.action_space == "Discrete":
                prob = self.actor(rollout_states)
                divKL = F.kl_div(old_log_prob.detach(), prob, reduction = 'batchmean')
                return divKL
            
            elif self.action_space == "Continuous":  
                distb = self.actor.Distb(rollout_states)
                old_mean = old_distb.mean.detach()
                old_cov = old_distb.covariance_matrix.sum(-1).detach()
                mean = distb.mean
                cov = distb.covariance_matrix.sum(-1)
                return (0.5)*((old_cov/cov).sum(-1)+(((old_mean - mean) ** 2)/cov).sum(-1)-self.action_dim + torch.log(cov).sum(-1) - torch.log(old_cov).sum(-1)).mean()
                   
        grad_kld_old_param = UATRPO.get_flat_grads(kld(), self.actor)
        
        def Hv(v):
            hessian_v = UATRPO.get_flat_grads(torch.dot(grad_kld_old_param, v), self.actor).detach()
            return hessian_v + self.conj_grad_damping*v
        
        gradient = UATRPO.get_flat_grads(L(), self.actor).detach()
        C = gradient - gradient.mean()
        C_reshaped = C.reshape(1,-1)
        n = C.shape[0]
        
        def Sigmav(v):
            return torch.tensordot(torch.tensordot(C_reshaped,v,1),C_reshaped,1)/(n-1)
        
        Y_F_cur = torch.zeros_like(self.Random_matrix)
        for i in range(self.random_projections):
            Y_F_cur[:,i] = Hv(self.Random_matrix[:,i])
            
        Y_F_new = self.beta*self.Y_F_old + (1-self.beta)*Y_F_cur
        Y_F = Y_F_new/(1-self.beta**(self.beta_t+1))
        
        Y_S_cur = torch.zeros_like(self.Random_matrix)
        for i in range(self.random_projections):
            Y_S_cur[:,i] = Sigmav(self.Random_matrix[:,i])
            
        Y_S_new = self.beta*self.Y_S_old + (1-self.beta)*Y_S_cur
        Y_S = Y_S_new/(1-self.beta**(self.beta_t+1))            
        
        Rn2 = (self.d + 2 * np.sqrt(self.d*np.log(1/self.alpha)) + 2*np.log(1/self.alpha)) / self.Total_trajs
        Y = Y_F + self.trade_off*Rn2*Y_S
        
        Q, sing, _ = np.linalg.svd(Y,full_matrices=False)
        Q_thresh = (np.finfo(Y.numpy().dtype).eps*np.max(sing)*np.max(Y.shape))
        Y_rank = np.sum(sing > Q_thresh)
        Q = Q[:,:Y_rank]
        self.Q = torch.FloatTensor(Q)
        
        def M(x):
            return Hv(x) + self.trade_off*Rn2*Sigmav(x)+self.conj_grad_damping*x
        
        if self.beta == 0:
            MQ = torch.zeros_like(self.Q)
            for i in range(Y_rank):
                MQ[:,i] = M(self.Q[:,i])  
            M_proj = torch.matmul(torch.transpose(self.Q,0,1),MQ)
        else:
            lstsq_goal = torch.matmul(torch.transpose(Y+self.conj_grad_damping*self.Random_matrix,0,1),self.Q)
            lstsq_input = torch.matmul(torch.transpose(self.Random_matrix,0,1),self.Q)
            M_proj = torch.transpose(torch.linalg.lstsq(lstsq_input, lstsq_goal).solution,0,1)
            
        D,V = torch.linalg.eigh(M_proj)
        V_thresh = (np.finfo(M_proj.numpy().dtype).eps*np.max(D.numpy())*np.max(M_proj.numpy().shape))
        M_proj_rank = np.sum(D.numpy() > V_thresh)
        D=D[-M_proj_rank:]
        V = V[:,-M_proj_rank:]
        
        U = torch.matmul(self.Q,V)
        g_proj = torch.tensordot(gradient, U, 1)
        v_proj = g_proj/D
        v_Qproj = torch.tensordot(V,v_proj,1)
        v_flat = torch.tensordot(U,v_proj,1)
        
        Mv = M(v_flat).detach()
        vMv = torch.tensordot(v_flat,Mv,1)
        eta = torch.sqrt((2 * self.epsilon)/vMv)
        eta_v_flat = eta*v_flat
        
        # Fv = Hv(eta_v_flat).detach()
        # Sv = Sigmav(eta_v_flat).detach()
        
        new_params, Failed = UATRPO.rescale_and_linesearch(self, gradient, eta_v_flat, _, _, L, Rn2, kld, old_params)

        if Entropy and not Failed:
            if self.action_space == "Discrete":
                _, entropy_log_prob = self.actor.sample_log(rollout_states, rollout_actions)
                discounted_casual_entropy = ((-1)*rollout_gammas*entropy_log_prob).mean()
                gradient_discounted_casual_entropy = UATRPO.get_flat_grads(discounted_casual_entropy, self.actor)
                new_params += self.lambda_*gradient_discounted_casual_entropy
            elif self.action_space == "Continuous":  
                discounted_casual_entropy = ((-1)*rollout_gammas*self.actor.Distb(rollout_states).log_prob(rollout_actions)).mean()
                gradient_discounted_casual_entropy = UATRPO.get_flat_grads(discounted_casual_entropy, self.actor)
                new_params += self.lambda_*gradient_discounted_casual_entropy
            
        UATRPO.set_params(self.actor, new_params)
        
        self.Y_F_old = Y_F_new
        self.Y_S_old = Y_S_new
        self.beta_t += 1
        
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))            
        
        
        
        
        
        
        

            
            
        
            
            
            

        