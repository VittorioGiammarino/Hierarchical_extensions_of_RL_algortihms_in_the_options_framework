#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:11:38 2021

@author: vittorio
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def H_Value_iteration(name_environment, Environment, nOptions, reward, offset_b=0, gamma=0.9, numIterations=100):
    
    P = Environment.P
    stateSpace = Environment.stateSpace

    nActions = Environment.action_size

    
    def Prob(P, initialPosition_ID, action):
        #this function returns a reward of rewardSize each step unless you are in a terminal
        #in that case, it returns a reward of zero
        #it returns the next state and reward    
        prob = P[initialPosition_ID, :, action]
        
        return prob
        
    #initialize value map to all zeros
    values_state_option = np.zeros((len(stateSpace), nOptions))
    values_state_option_action = np.zeros((len(stateSpace), nActions, nOptions))
    
    deltas = []
    
    for it in range(numIterations):
        
        #make a copy of the value function to manipulate during the algorithm
        copyValue = np.copy(values_state_option)
        copyValue_actions = np.copy(values_state_option_action)
        
        deltaState = []
        #this will be set to Vcurrent - Vnext
        for state in range(len(stateSpace)):
            for option in range(nOptions):
            
                #the next variable will be equal to the new V iterate by the end of the process
                weightedRewards = 0
                tempV = []
                #Compute the Bellman iterate
                for action in range(5):
                    #compute next position and reward from taking that action
                    prob = Prob(P, state, action)
                
                    #since V_next = \sum_{actions} \pi(a|s) \sum_{s’} p(s’|a,s)[r + gamma*V]
                    #we can just go through the sum and add each element to weightedRewards
                    #we use here that next state is deterministic
                    
                    tempV.append(reward[state,option,action]+gamma*np.sum(prob*values_state_option[:,option]))
                    
                    
                weightedRewards = max(tempV)
                deltaState.append(np.abs(values_state_option[state,option]-weightedRewards))
                values_state_option_action[state,:,option] = np.array(tempV)
                
            
                #update the value of the next state, but in the copy rather than the original
                copyValue[state,option] = weightedRewards
            
        #this is now an array of size numIterations, where every entry is an array of Vcurre
        deltas.append(deltaState)
        
        #update the value map with what we just computed
        values_state_option = copyValue
        
    pi_lo = np.zeros((len(stateSpace),nOptions))
    for option in range(nOptions):
        pi_lo[:,option] = np.argmax(values_state_option_action[:,:,option],1)
        
    plt.figure(figsize=(20, 10))
    plt.plot(deltas)
    
    if not os.path.exists("./Figures/HVI"):
            os.makedirs("./Figures/HVI")
    
    plt.savefig(f"Figures/HVI/Value_iteration_{name_environment}.svg", format = 'svg')
        
    pi_hi = (1/nOptions)*np.ones((len(stateSpace),nOptions))
    pi_b = np.zeros((len(stateSpace),nOptions))
    
    
    for _ in range(100):
        for state in range(len(stateSpace)): 
            for option in range(nOptions):
                
                action = int(pi_lo[state,option])
                prob = Prob(P, state, action)
                tempV_termination = []
                
                for termination in range(2):
                    if termination == 0:
                        tempV_termination.append(reward[state,option,action]+gamma*np.sum(prob*values_state_option[:,option]))   
                    else:
                        tempV_termination.append(reward[state,option,action]+gamma*(np.sum(prob*(np.sum(pi_hi*values_state_option,1)))-offset_b))  
                                                                                            
                pi_b[state,option] = np.argmax(np.array(tempV_termination))
                
        for state in range(len(stateSpace)): 
            
            tempV_pi_hi = []
            for option in range(nOptions):
                
                action = int(pi_lo[state,option])
                prob = Prob(P, state, action)
                
                tempV_pi_hi.append(reward[state,option,action]+gamma*np.sum(prob*values_state_option[:,option]))  
        
            pi_hi_greedy = np.argmax(np.array(tempV_pi_hi))
            
            pi_hi[state, pi_hi_greedy] = 1
            pi_hi[state, 1-pi_hi_greedy] = 0
            
    return pi_lo, pi_hi, pi_b, values_state_option