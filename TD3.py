import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from Buffer import ReplayBuffer
from models import DeepDeterministicHierarchicalActor
from models import SoftmaxHierarchicalActor
from models import Critic_flat
from models import Critic_flat_discrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class TD3(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action,
              l_rate_actor=3e-4, l_rate_critic=3e-4, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        if np.isinf(action_space_cardinality):
            self.actor = DeepDeterministicHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Continuous"
            
            self.critic = Critic_flat(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_critic)
        else:
            self.actor = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            self.actor_target = copy.deepcopy(self.actor)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
            self.action_space = "Discrete"
                    
            self.critic = Critic_flat_discrete(state_dim, action_space_cardinality).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_critic)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action

        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        self.Buffer = ReplayBuffer(state_dim, action_dim)

        self.total_it = 0

    def select_action(self, state):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            prob_u = self.actor(state).cpu().data.numpy()
            action = np.argmax(prob_u)
            return int(action)
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.actor(state).cpu().data.numpy().flatten()
        
    def explore(self, state, expl_noise):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            prob_u = self.actor(state).cpu().data.numpy()
            noised_prob = prob_u + np.random.normal(0, expl_noise, size=self.action_space_cardinality)
            prob_u = np.exp(noised_prob) / np.sum(np.exp(noised_prob))
            prob_u = torch.FloatTensor(prob_u)
            m = Categorical(prob_u)
            action = m.sample()            
            return int(action.detach().data.numpy().flatten())
            
        if self.action_space == "Continuous":
            state = np.array(state)
            action = (TD3.select_action(self, state) + np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
            return action

    def train(self, batch_size=256):
        self.total_it += 1

		# Sample replay buffer 
        state, action, next_state, reward, cost, not_done = self.Buffer.sample(batch_size)

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            
            if self.action_space == "Discrete":
                noise = (torch.randn_like(torch.zeros((batch_size, self.action_space_cardinality))) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                normalize = nn.Softmax(dim=1)
                next_action_prob = normalize(self.actor_target(next_state) + noise)
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state)
                target_Q = next_action_prob*(torch.min(target_Q1, target_Q2))
                target_Q = reward-cost + not_done * self.discount * target_Q.sum(dim=1).unsqueeze(-1)
                
            elif self.action_space == "Continuous":    
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action[0], self.max_action[0])
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.discount * target_Q

        if self.action_space == "Discrete":
            Q1, Q2 = self.critic(state)
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            self.actor.train()
            if self.action_space == "Discrete":
                Q1, Q2 = self.critic(state)
                minQ = torch.min(Q1,Q2)
                action_prob = self.actor(state)
                
                actor_loss = -((action_prob*minQ).sum(dim=1)).mean()
            
            elif self.action_space == "Continuous":

    			# Compute actor losse
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			
			# Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
        
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
        self.actor_target = copy.deepcopy(self.actor)
        
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
		