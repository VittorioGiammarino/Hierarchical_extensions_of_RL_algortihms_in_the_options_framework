import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from Buffer import ReplayBuffer
from models import DeepDeterministicHierarchicalActor
from models import SoftmaxHierarchicalActor
from models import Critic
from models import Critic_discrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class H_TD3(object):
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, option_dim, termination_dim, 
                 l_rate_pi_lo=3e-4, l_rate_pi_hi=3e-4 , l_rate_pi_b=3e-4, l_rate_critic = 3e-4, 
                 discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,  eta = 0.001, pi_lo_freq=2, pi_b_freq=5, pi_hi_freq=5):
        
        self.Buffer = [[None]*1 for _ in range(option_dim)]
        self.pi_lo = [[None]*1 for _ in range(option_dim)]
        self.pi_b = [[None]*1 for _ in range(option_dim)]
        self.pi_lo_target = [[None]*1 for _ in range(option_dim)]
        self.pi_b_target = [[None]*1 for _ in range(option_dim)]
        
        if np.isinf(action_space_cardinality):
            self.pi_hi = DeepDeterministicHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            self.pi_hi_target = copy.deepcopy(self.pi_hi)
            pi_lo_temp = DeepDeterministicHierarchicalActor.NN_PI_LO(state_dim, action_dim, max_action).to(device)
            pi_b_temp = DeepDeterministicHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            self.action_space = "Continuous"
            
            self.critic = Critic(state_dim, action_dim, option_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)
        else:
            self.pi_hi = SoftmaxHierarchicalActor.NN_PI_HI(state_dim, option_dim).to(device)
            self.pi_hi_target = copy.deepcopy(self.pi_hi)
            pi_lo_temp = SoftmaxHierarchicalActor.NN_PI_LO(state_dim, action_space_cardinality).to(device)
            pi_b_temp = SoftmaxHierarchicalActor.NN_PI_B(state_dim, termination_dim).to(device)
            self.action_space = "Discrete"
                    
            self.critic = Critic_discrete(state_dim, action_space_cardinality, option_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic)    
        
        for option in range(option_dim):
            self.Buffer[option] = ReplayBuffer(state_dim, action_dim)
            self.pi_lo[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b[option] = copy.deepcopy(pi_b_temp)        
            self.pi_lo_target[option] = copy.deepcopy(pi_lo_temp)
            self.pi_b_target[option] = copy.deepcopy(pi_b_temp)  
            
        # define optimizer 
        self.pi_hi_optimizer = torch.optim.Adam(self.pi_hi.parameters(), lr=l_rate_pi_hi)
        self.pi_b_optimizer = [[None]*1 for _ in range(option_dim)] 
        self.pi_lo_optimizer = [[None]*1 for _ in range(option_dim)]  
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_critic)  
        for option in range(option_dim):
            self.pi_lo_optimizer[option] = torch.optim.Adam(self.pi_lo[option].parameters(), lr=l_rate_pi_lo)
            self.pi_b_optimizer[option] = torch.optim.Adam(self.pi_b[option].parameters(), lr=l_rate_pi_b)  

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        self.option_dim = option_dim
        self.termination_dim = termination_dim

        self.discount = discount
        self.tau = tau
        self.eta = eta
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.pi_lo_freq = pi_lo_freq
        self.pi_b_freq = pi_b_freq
        self.pi_hi_freq = pi_hi_freq
        self.learning_rate_pi_lo = l_rate_pi_lo
        self.learning_rate_pi_b = l_rate_pi_b
        self.learning_rate_pi_hi = l_rate_pi_hi

        self.total_it = 0

    def select_action(self, state, option):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            prob_u = self.pi_lo[option](state).cpu().data.numpy()
            action = np.argmax(prob_u)
            return int(action)
        
        if self.action_space == "Continuous":
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            return self.pi_lo[option](state).cpu().data.numpy().flatten()
                
    def explore(self, state, option, expl_noise):
        if self.action_space == "Discrete":
            state = torch.FloatTensor(state.reshape(1,-1)).to(device)
            prob_u = self.pi_lo[option](state).cpu().data.numpy()
            noised_prob = prob_u + np.random.normal(0, expl_noise, size=self.action_space_cardinality)
            prob_u = np.exp(noised_prob) / np.sum(np.exp(noised_prob))
            prob_u = torch.FloatTensor(prob_u)
            m = Categorical(prob_u)
            action = m.sample()            
            return int(action.detach().data.numpy().flatten())
            
        if self.action_space == "Continuous":
            state = np.array(state)
            action = (H_TD3.select_action(self, state, option) + np.random.normal(0, self.max_action * expl_noise, size=self.action_dim)).clip(-self.max_action, self.max_action)
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

    def train(self, option, batch_size=256):
        
        self.total_it += 1

		# Sample replay buffer 
        state, action, next_state, reward, cost, not_done = self.Buffer[option].sample(batch_size)
        option_vector = torch.ones_like(reward[:,0] , dtype=int)

        with torch.no_grad():
			# Select action according to policy and add clipped noise
            
            if self.action_space == "Discrete":
                
                noise = (torch.randn_like(torch.zeros((batch_size, self.action_space_cardinality))) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                normalize = nn.Softmax(dim=1)
                
                next_action_target_prob_option_i = []
                for option_i in range(self.option_dim): 
                    next_action_target_prob = normalize(self.pi_lo_target[option_i](next_state) + noise)
                    next_action_target_prob_option_i.append(next_action_target_prob)
                    
                next_action_target_prob = next_action_target_prob_option_i[option]
                first_term_target_Q1, first_term_target_Q2 = self.critic_target(next_state, F.one_hot(option*option_vector, num_classes=self.option_dim))
                first_term_target_Q1 = (next_action_target_prob*first_term_target_Q1).sum(dim=1).unsqueeze(-1)
                first_term_target_Q2 = (next_action_target_prob*first_term_target_Q2).sum(dim=1).unsqueeze(-1)
                pi_b_target = self.pi_b_target[option](next_state)
                target_Q1 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q1
                target_Q2 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q2                
                
                for option_i in range(self.option_dim):
                    next_action_target_prob = next_action_target_prob_option_i[option_i]
                    second_term_target_Q1, second_term_target_Q2 = self.critic_target(next_state, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    second_term_target_Q1 = (next_action_target_prob*second_term_target_Q1).sum(dim=1).unsqueeze(-1)
                    second_term_target_Q2 = (next_action_target_prob*second_term_target_Q2).sum(dim=1).unsqueeze(-1)
                    target_Q1 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q1
                    target_Q2 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q2                  
                
                target_Q = torch.min(target_Q1, target_Q2) 
                target_Q = reward-cost + not_done * self.discount * target_Q                
                
            elif self.action_space == "Continuous":    
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                next_action_target_option_i = []
                for option_i in range(self.option_dim):
                    next_action_target = self.pi_lo_target[option_i](next_state)
                    next_action_target_option_i.append(((next_action_target + noise).clamp(-self.max_action[0], self.max_action[0])).to(device))
                
                next_action_target = next_action_target_option_i[option]
                first_term_target_Q1, first_term_target_Q2 = self.critic_target(next_state, next_action_target, F.one_hot(option*option_vector, num_classes=self.option_dim))
                pi_b_target = self.pi_b_target[option](next_state)
                target_Q1 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q1
                target_Q2 = pi_b_target[:,0].reshape(-1,1)*first_term_target_Q2
                
                for option_i in range(self.option_dim):
                    next_action_target = next_action_target_option_i[option_i]
                    second_term_target_Q1, second_term_target_Q2 = self.critic_target(next_state, next_action_target, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    target_Q1 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q1
                    target_Q2 += pi_b_target[:,1].reshape(-1,1)*self.pi_hi_target(next_state)[:,option_i].reshape(-1,1)*second_term_target_Q2
                     
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward-cost + not_done * self.discount * target_Q                

        if self.action_space == "Discrete":
            Q1, Q2 = self.critic(state, F.one_hot(option*option_vector, num_classes=self.option_dim))
            current_Q1 = Q1.gather(1, action.detach().long()) 
            current_Q2 = Q2.gather(1, action.detach().long()) 
        
        elif self.action_space == "Continuous":
            #current Q estimates
            current_Q1, current_Q2 = self.critic(state, action, F.one_hot(option*option_vector, num_classes=self.option_dim))

		# Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

		# Delayed policy updates actor-critic
        if self.total_it % self.pi_lo_freq == 0:

            self.pi_lo[option].train()
            if self.action_space == "Discrete":
                Q1, Q2 = self.critic(state, F.one_hot(option*option_vector, num_classes=self.option_dim))
                minQ = torch.min(Q1,Q2)
                action_prob = self.pi_lo[option](state)
                
                pi_lo_loss = -((action_prob*minQ).sum(dim=1)).mean()
            
            elif self.action_space == "Continuous":
    			# Compute actor losse
                action = self.pi_lo[option](state)
                pi_lo_loss = -(self.critic.Q1(state, action, F.one_hot(option*option_vector, num_classes=self.option_dim))).mean()
			
            # Optimize pi_lo 
            self.pi_lo_optimizer[option].zero_grad()
            pi_lo_loss.backward()
            self.pi_lo_optimizer[option].step()

			# Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.pi_lo[option].parameters(), self.pi_lo_target[option].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
		# Delayed policy updates pi_b
        if self.total_it % self.pi_b_freq == 0:
            
            pi_b = self.pi_b[option](next_state)
            
            if self.action_space == "Discrete":
                
                next_action_prob_option_i = []
                for option_i in range(self.option_dim): 
                    next_action_prob = normalize(self.pi_lo[option_i](next_state) + noise)
                    next_action_prob_option_i.append(next_action_prob)
                
                next_action_prob = next_action_prob_option_i[option]
                Q1, Q2 = self.critic(next_state, F.one_hot(option*option_vector, num_classes=self.option_dim))
                Q1 = (next_action_prob*Q1).sum(dim=1).unsqueeze(-1)
                Q2 = (next_action_prob*Q2).sum(dim=1).unsqueeze(-1)
                pi_b_loss = pi_b[:,1].reshape(-1,1)*(torch.min(Q1,Q2) + self.eta)
                
                for option_i in range(self.option_dim):
                    next_action_prob = next_action_prob_option_i[option_i]
                    Q1, Q2 = self.critic(next_state, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    Q1 = (next_action_prob*Q1).sum(dim=1).unsqueeze(-1)
                    Q2 = (next_action_prob*Q2).sum(dim=1).unsqueeze(-1)
                    pi_b_loss -= pi_b[:,1].reshape(-1,1)*self.pi_hi(next_state)[:,option_i].reshape(-1,1)*torch.min(Q1,Q2)
  
            elif self.action_space == "Continuous":         

                next_action_option_i = []
                for option_i in range(self.option_dim):
                    next_action = self.pi_lo[option_i](next_state)
                    next_action_option_i.append(((next_action + noise).clamp(-self.max_action[0], self.max_action[0])).to(device))

                next_action = next_action_option_i[option]
                Q1, Q2 = self.critic(next_state, next_action, F.one_hot(option*option_vector, num_classes=self.option_dim))
                pi_b_loss = pi_b[:,1].reshape(-1,1)*(torch.min(Q1,Q2) + self.eta)
                for option_i in range(self.option_dim):
                    next_action = next_action_option_i[option_i]
                    Q1, Q2 = self.critic(next_state, next_action, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    pi_b_loss -= pi_b[:,1].reshape(-1,1)*self.pi_hi(next_state)[:,option_i].reshape(-1,1)*torch.min(Q1,Q2)
                           
            # Optimize pi_lo 
            pi_b_loss = torch.mean(pi_b_loss) #(-1)*torch.mean((-1)*pi_b_loss)
            self.pi_b_optimizer[option].zero_grad()
            pi_b_loss.backward()
            self.pi_b_optimizer[option].step()
            
			# Update the frozen target models    
            for param, target_param in zip(self.pi_b[option].parameters(), self.pi_b_target[option].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        if self.total_it % self.pi_hi_freq == 0:
            # Compute pi_hi loss

            if self.action_space == "Discrete":
                
                next_action_prob_option_i = []
                for option_i in range(self.option_dim): 
                    next_action_prob = normalize(self.pi_lo[option_i](next_state) + noise)
                    next_action_prob_option_i.append(next_action_prob)
                
                next_action_prob = next_action_prob_option_i[option]
                Q1, Q2 = self.critic(next_state, F.one_hot(option*option_vector, num_classes=self.option_dim))
                Q1 = (next_action_prob*Q1).sum(dim=1).unsqueeze(-1)
                Q2 = (next_action_prob*Q2).sum(dim=1).unsqueeze(-1)
                pi_hi_loss = torch.min(Q1,Q2)
                
                for option_i in range(self.option_dim):
                    next_action_prob = next_action_prob_option_i[option_i]
                    Q1, Q2 = self.critic(next_state, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    Q1 = (next_action_prob*Q1).sum(dim=1).unsqueeze(-1)
                    Q2 = (next_action_prob*Q2).sum(dim=1).unsqueeze(-1)
                    pi_hi_loss -= self.pi_hi(next_state)[:,option_i].reshape(-1,1)*torch.min(Q1,Q2)             
            
            elif self.action_space == "Continuous":
                
                next_action_option_i = []
                for option_i in range(self.option_dim):
                    next_action = self.pi_lo[option_i](next_state)
                    next_action_option_i.append(((next_action + noise).clamp(-self.max_action[0], self.max_action[0])).to(device))
                
                next_action = next_action_option_i[option]
                Q1, Q2 = self.critic(next_state, next_action, F.one_hot(option*option_vector, num_classes=self.option_dim))
                pi_hi_loss = torch.min(Q1,Q2)
                for option_i in range(self.option_dim):
                    next_action = next_action_option_i[option_i]
                    Q1, Q2 = self.critic(next_state, next_action, F.one_hot(option_i*option_vector, num_classes=self.option_dim))
                    pi_hi_loss -= self.pi_hi(next_state)[:,option_i].reshape(-1,1)*torch.min(Q1,Q2)
                
            pi_hi_loss = -torch.mean(torch.log((self.pi_hi(next_state)[:,option].reshape(-1,1)).clamp(1e-10,1))*pi_hi_loss)
            
            self.pi_hi_optimizer.zero_grad()
            pi_hi_loss.backward()
            self.pi_hi_optimizer.step()
            
            for param, target_param in zip(self.pi_hi.parameters(), self.pi_hi_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

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
        self.pi_hi_target.copy.deepcopy(self.pi_hi)
        
        for option in range(self.option_dim):
            self.pi_lo[option].load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
            self.pi_lo_optimizer[option].load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))
            self.pi_lo_target[option].copy.deepcopy(self.pi_lo[option])
            self.pi_b[option].load_state_dict(torch.load(filename + f"_pi_b_option_{option}"))
            self.pi_b_optimizer[option].load_state_dict(torch.load(filename + f"_pi_b_optimizer_option_{option}"))
            self.pi_b_target[option].copy.deepcopy(self.pi_b[option])
            
    def save_critic(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
    
    def load_critic(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
		