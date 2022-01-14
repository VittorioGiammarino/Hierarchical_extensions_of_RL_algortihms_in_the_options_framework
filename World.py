#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 13:34:00 2020

@author: vittorio
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class TwoRooms:
    class Environment:
        def __init__(self, reward_coordinate = np.array([[10,10], [0,0]]), reward_sequence = np.array([0,1]), init_state = np.array([0,0]), max_episode_steps = 300):
            self.Nc = 10 #Time steps required to bring drone to base when it crashes
            self.P_WIND = 0.1 #Gust of wind probability
            #IDs of elements in map
            self.FREE = 0
            self.WALL = 1

            #Actions index
            self.NORTH = 0
            self.SOUTH = 1
            self.EAST = 2
            self.WEST = 3
            self.HOVER = 4
            
            self.state = init_state
            self.init_state = init_state
            self.observation_space = np.array([len(self.state)])
            
            self.action_dim = 1
            self.action_size = 5
            self._max_episode_steps = max_episode_steps
            self.step_counter = 0
            
            self.reward_coordinate = reward_coordinate
            self.reward_sequence = reward_sequence
            
            TwoRooms.Environment.GenerateMap_StateSpace(self)
            TwoRooms.Environment.ComputeTransitionProbabilityMatrix(self)
            
        def GenerateMap_StateSpace(self):    
            mapsize = [5, 11]
            grid = np.zeros((mapsize[0], mapsize[1]))
        
            #define obstacles
            grid[0:2,5] = self.WALL
            grid[3:,5]= self.WALL

            self.map = grid
            
            stateSpace = np.empty((0,2),int)
            for m in range(0,self.map.shape[0]):
                for n in range(0,self.map.shape[1]):
                    if self.map[m,n] != self.WALL:
                        stateSpace = np.append(stateSpace, [[m, n]], 0)
                    
            self.stateSpace = stateSpace


        def FindStateIndex(self, value):
    
            K = self.stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if self.stateSpace[k,0]==value[0] and self.stateSpace[k,1]==value[1]:
                    stateIndex = k
    
            return stateIndex
        
        def ComputeTransitionProbabilityMatrix(self):
            action_space=5
            K = self.stateSpace.shape[0]
            P = np.zeros((K,K,action_space))
            [M,N]=self.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = TwoRooms.Environment.FindStateIndex(self,array_temp)

                    if self.map[i,j] != self.WALL:

                        for u in range(0,action_space):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == self.EAST and self.map[i,j+1]!=self.WALL:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                                elif j==N-1 and u==self.EAST:
                                    comp_no=1
                            #west case
                            if j!=0:
                                if u==self.WEST and self.map[i,j-1]!=self.WALL:
                                    r=i
                                    s=j-1
                                    comp_no=0
                                elif j==0 and u==self.WEST:
                                    comp_no=1
                            #south case
                            if i!=0:
                                if u==self.SOUTH and self.map[i-1,j]!=self.WALL:
                                    r=i-1
                                    s=j
                                    comp_no=0
                                elif i==0 and u==self.SOUTH:
                                    comp_no=1
                            #north case
                            if i!=M-1:
                                if u==self.NORTH and self.map[i+1,j]!=self.WALL:
                                    r=i+1
                                    s=j
                                    comp_no=0
                                elif i==M-1 and u==self.NORTH:
                                    comp_no=1
                            #hover case
                            if u==self.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t = TwoRooms.Environment.FindStateIndex(self,array_temp)

                                # No wind case
                                P[k,t,u] = P[k,t,u]+(1-self.P_WIND)

                                # case wind

                                #north wind
                                if s+1>N-1 or self.map[r,s+1]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t = TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #north wind no hit

                                #South Wind
                                if s-1<0 or self.map[r,s-1]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)                                 
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #south wind no hit

                                #East Wind
                                if r+1>M-1 or self.map[r+1,s]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #east wind no hit

                                #West Wind
                                if r-1<0 or self.map[r-1,s]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=TwoRooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #west wind no hit

                            elif comp_no == 1:
                                base0=TwoRooms.Environment.FindStateIndex(self, self.init_state)
                                P[k,k,u]=1

            self.P = P
        
        def PlotMap(self, name):
            mapsize = self.map.shape
            #count walls
            nwalls=0;
            walls = np.empty((0,2),int)
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.map[i,j]==self.WALL:
                        walls = np.append(walls, [[j, i]], 0)
                        nwalls += 1
                        
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    
            for i in range(0,nwalls):
                plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
        
            for i in range(0,nwalls):
                plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.draw()
            plt.savefig(name +'.pdf', format='pdf')
            
        def Seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
                
        def reset(self, version = 'standard', init_state = np.array([0,0,0])):
            if version == 'standard':
                self.state = init_state
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
            else:
                mapsize = self.map.shape()
                for i in range(30):
                    init_state = np.random.randint(0,mapsize[1],2)
                    if self.map[init_state] == self.FREE:
                        break
                self.state = np.append(init_state,0)
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
                
            return self.state
                    
        def random_sample(self):
            return np.random.randint(0,self.action_size)       
        
        def step(self, action):
            
            self.step_counter +=1
            r=0
            
            # given action, draw next state
            state_index = Four_Rooms.Environment.FindStateIndex(self, self.state[0:2])
            x_k_possible=np.where(self.P[state_index,:,int(action)]!=0)
            prob = self.P[state_index,x_k_possible[0][:],int(action)]
            prob_rescaled = np.divide(prob,np.amin(prob))

            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            state_plus1 = self.stateSpace[x_k_possible[0][index_x_plus1],:]
            
            if state_plus1[0] == self.reward_coordinate[self.active_reward][0] and state_plus1[1] == self.reward_coordinate[self.active_reward][1]:
                r=r+1
                if self.reward_counter < len(self.reward_sequence)-1:
                    self.reward_counter += 1
                else:
                    self.reward_counter = 0
                    
                self.active_reward = self.reward_sequence[self.reward_counter]
                
            self.state = np.append(state_plus1, self.active_reward)
            if self.step_counter >= self._max_episode_steps:
                done = True
            else:
                done = False            
            
            return self.state, r, done, False  
        
        
class Four_Rooms:
    class Environment:
        def __init__(self, reward_coordinate = np.array([[10,10], [0,0]]), reward_sequence = np.array([0,1]), init_state = np.array([0,0,0]), max_episode_steps = 300):
            self.Nc = 30 #Time steps required to bring drone to base when it crashes
            self.P_WIND = 0.2 #Gust of wind probability
            self.FREE = 0
            self.WALL = 1
            
            #Actions index
            self.NORTH = 0
            self.SOUTH = 1
            self.EAST = 2
            self.WEST = 3
            self.HOVER = 4
            
            self.state = init_state
            self.init_state = init_state
            self.observation_space = np.array([len(self.state)])
            
            self.action_dim = 1
            self.action_size = 5
            self._max_episode_steps = max_episode_steps
            self.step_counter = 0
            
            self.reward_coordinate = reward_coordinate
            self.reward_sequence = reward_sequence
            
            Four_Rooms.Environment.GenerateMap_StateSpace(self)
            Four_Rooms.Environment.ComputeTransitionProbabilityMatrix(self)
            
        def GenerateMap_StateSpace(self):    
            mapsize = [11, 11]
            grid = np.zeros((mapsize[0], mapsize[1]))
            #define obstacles
            grid[0,5] = self.WALL
            grid[2:8,5]= self.WALL
            grid[9:11,5]= self.WALL
            grid[5,0]= self.WALL
            grid[5,2:5]= self.WALL
            grid[4,6:8]= self.WALL
            grid[4,9:11]= self.WALL
                        
            self.map = grid
            
            stateSpace = np.empty((0,2),int)
            for m in range(0,self.map.shape[0]):
                for n in range(0,self.map.shape[1]):
                    if self.map[m,n] != self.WALL:
                        stateSpace = np.append(stateSpace, [[m, n]], 0)
                    
            self.stateSpace = stateSpace
                        
        def FindStateIndex(self, value):
            K = self.stateSpace.shape[0];
            stateIndex = 0
    
            for k in range(0,K):
                if self.stateSpace[k,0]==value[0] and self.stateSpace[k,1]==value[1]:
                    stateIndex = k
    
            return stateIndex
        
        def ComputeTransitionProbabilityMatrix(self):
            K = self.stateSpace.shape[0]
            P = np.zeros((K,K,self.action_size))
            [M,N]=self.map.shape

            for i in range(0,M):
                for j in range(0,N):

                    array_temp = [i, j]
                    k = Four_Rooms.Environment.FindStateIndex(self,array_temp)

                    if self.map[i,j] != self.WALL:

                        for u in range(0,self.action_size):
                            comp_no=1;
                            # east case
                            if j!=N-1:
                                if u == self.EAST and self.map[i,j+1]!=self.WALL:
                                    r=i
                                    s=j+1
                                    comp_no = 0
                                elif j==N-1 and u==self.EAST:
                                    comp_no=1
                            #west case
                            if j!=0:
                                if u==self.WEST and self.map[i,j-1]!=self.WALL:
                                    r=i
                                    s=j-1
                                    comp_no=0
                                elif j==0 and u==self.WEST:
                                    comp_no=1
                            #south case
                            if i!=0:
                                if u==self.SOUTH and self.map[i-1,j]!=self.WALL:
                                    r=i-1
                                    s=j
                                    comp_no=0
                                elif i==0 and u==self.SOUTH:
                                    comp_no=1
                            #north case
                            if i!=M-1:
                                if u==self.NORTH and self.map[i+1,j]!=self.WALL:
                                    r=i+1
                                    s=j
                                    comp_no=0
                                elif i==M-1 and u==self.NORTH:
                                    comp_no=1
                            #hover case
                            if u==self.HOVER:
                                r=i
                                s=j
                                comp_no=0

                            if comp_no==0:
                                array_temp = [r, s]
                                t = Four_Rooms.Environment.FindStateIndex(self, array_temp)
 
                                # No wind case
                                P[k,t,u] = P[k,t,u]+(1-self.P_WIND)
                                base0 = Four_Rooms.Environment.FindStateIndex(self, self.init_state)

                                # case wind

                                #north wind
                                if s+1>N-1 or self.map[r,s+1]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s+1]
                                    t = Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #north wind no hit

                                #South Wind
                                if s-1<0 or self.map[r,s-1]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r, s-1]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)                                 
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #south wind no hit

                                #East Wind
                                if r+1>M-1 or self.map[r+1,s]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r+1, s]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #east wind no hit

                                #West Wind
                                if r-1<0 or self.map[r-1,s]==self.WALL:
                                    P[k,k,u]=P[k,k,u]+self.P_WIND*0.25 #wind causes crash
                                else:
                                    array_temp = [r-1, s]
                                    t=Four_Rooms.Environment.FindStateIndex(self,array_temp)
                                    P[k,t,u] = P[k,t,u]+0.25*self.P_WIND #west wind no hit

                            elif comp_no == 1:
                                base0 = Four_Rooms.Environment.FindStateIndex(self, self.init_state)
                                P[k,k,u]=1

            self.P = P
            
            
        def PlotMap(self, name):
            mapsize = self.map.shape
            #count walls
            nwalls=0;
            walls = np.empty((0,2),int)
            for i in range(0,mapsize[0]):
                for j in range(0,mapsize[1]):
                    if self.map[i,j]==self.WALL:
                        walls = np.append(walls, [[j, i]], 0)
                        nwalls += 1
                        
            plt.figure()
            plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
    
            for i in range(0,nwalls):
                plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
        
            for i in range(0,nwalls):
                plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                         [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.draw()
            plt.savefig(name +'.pdf', format='pdf')
                
                  
        def Seed(self, seed):
            self.seed = seed
            np.random.seed(self.seed)
                
        def reset(self, version = 'standard', init_state = np.array([0,0,0])):
            if version == 'standard':
                self.state = init_state
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
            else:
                mapsize = self.map.shape()
                for i in range(30):
                    init_state = np.random.randint(0,mapsize[1],2)
                    if self.map[init_state] == self.FREE:
                        break
                self.state = np.append(init_state,0)
                self.step_counter = 0
                self.reward_counter = 0
                self.active_reward = int(self.state[2])
                
            return self.state
                    
        def random_sample(self):
            return np.random.randint(0,self.action_size)       
        
        def step(self, action):
            
            self.step_counter +=1
            r=0
            
            # given action, draw next state
            state_index = Four_Rooms.Environment.FindStateIndex(self, self.state[0:2])
            x_k_possible=np.where(self.P[state_index,:,int(action)]!=0)
            prob = self.P[state_index,x_k_possible[0][:],int(action)]
            prob_rescaled = np.divide(prob,np.amin(prob))

            for i in range(1,prob_rescaled.shape[0]):
                prob_rescaled[i]=prob_rescaled[i]+prob_rescaled[i-1]
            draw=np.divide(np.random.rand(),np.amin(prob))
            index_x_plus1=np.amin(np.where(draw<prob_rescaled))
            state_plus1 = self.stateSpace[x_k_possible[0][index_x_plus1],:]
            
            if state_plus1[0] == self.reward_coordinate[self.active_reward][0] and state_plus1[1] == self.reward_coordinate[self.active_reward][1]:
                r=r+1
                if self.reward_counter < len(self.reward_sequence)-1:
                    self.reward_counter += 1
                else:
                    self.reward_counter = 0
                    
                self.active_reward = self.reward_sequence[self.reward_counter]
                
            self.state = np.append(state_plus1, self.active_reward)
            if self.step_counter >= self._max_episode_steps:
                done = True
            else:
                done = False            
            
            return self.state, r, done, False  
                
class Simulation:
    def __init__(self, environment, name_env):
        self.Environment = environment
        self.name_env = name_env
    
    def VideoSimulation(self, u, states, reward_location, name_video):
        
        mapsize = self.Environment.map.shape
        nwalls=0;
        walls = np.empty((0,2),int)
        for i in range(0,mapsize[0]):
            for j in range(0,mapsize[1]):
                if self.Environment.map[i,j]==self.Environment.WALL:
                    walls = np.append(walls, [[j, i]], 0)
                    nwalls += 1        
        # Plot
        fig = plt.figure()
        plt.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')

        for i in range(0,nwalls):
            plt.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                     [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
    
        for i in range(0,nwalls):
            plt.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                     [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
            
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        plt.draw()

        ims = []
        for s in range(0,len(u)):
            if u[s] == self.Environment.NORTH:
                txt = u'\u2191'
            elif u[s] == self.Environment.SOUTH:
                txt = u'\u2193'
            elif u[s] == self.Environment.EAST:
                txt = u'\u2192'
            elif u[s] == self.Environment.WEST:
                txt = u'\u2190'
            elif u[s] == self.Environment.HOVER:
                txt = u'\u2715'    
            im1 = plt.text(states[s,1]+0.0, states[s,0]+0.0, txt, fontsize=20)
            im2 = plt.text(reward_location[s,1]+0.0, reward_location[s,0]+0.0, 'R', fontsize=20, color = 'r')
            ims.append([im1, im2])

        ani = animation.ArtistAnimation(fig, ims, interval=1200, blit=True,
                                        repeat_delay=2000)
        
        if not os.path.exists(f"./Videos/{self.name_env}"):
            os.makedirs(f"./Videos/{self.name_env}")
        
        ani.save(f"./Videos/{self.name_env}/" + name_video)
            
            
            
            
            