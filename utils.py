#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:00:31 2021

@author: vittorio
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def Plot_summary(name_env, Environment, pi_lo, pi_hi, pi_b, values_state_option, size = (18,10)):
    columns = 4
    rows = 2
    fig, ax_array = plt.subplots(rows, columns, squeeze=False, figsize=size)
    i=0
    for k,ax_row in enumerate(ax_array):
        for n_row, axes in enumerate(ax_row):
            
            l = n_row%2
            
            if k==0 and n_row<2:
                mapsize = Environment.map.shape
                nwalls=0;
                walls = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if Environment.map[i,j]==Environment.WALL:
                            walls = np.append(walls, [[j, i]], 0)
                            nwalls += 1        
                # Plot
                axes.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
        
                for i in range(0,nwalls):
                    axes.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
            
                for i in range(0,nwalls):
                    axes.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                    
                    
                u = pi_lo[:,l]
                for s in range(0,u.shape[0]):
                    if u[s] == Environment.NORTH:
                        txt = u'\u2191'
                    elif u[s] == Environment.SOUTH:
                        txt = u'\u2193'
                    elif u[s] == Environment.EAST:
                        txt = u'\u2192'
                    elif u[s] == Environment.WEST:
                        txt = u'\u2190'
                    elif u[s] == Environment.HOVER:
                        txt = u'\u2715'
                    axes.text(Environment.stateSpace[s,1]+0.3, Environment.stateSpace[s,0]+0.5, txt)
                    
                axes.title.set_text(f"low level option {l}")    
                axes.axis('off')
                
            elif k==0 and n_row>=2:
                if l == 0:
                    color = 'y'
                if l == 1:
                    color = 'r'
                    
                mapsize = Environment.map.shape
                #count walls
                nwalls=0;
                walls = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if Environment.map[i,j]==Environment.WALL:
                            walls = np.append(walls, [[j, i]], 0)
                            nwalls += 1
                            
                axes.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
        
                for i in range(0,nwalls):
                    axes.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
            
                for i in range(0,nwalls):
                    axes.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                    
                distribution = pi_hi[:,l]    
                for s in range(0,len(Environment.stateSpace)):        
                    axes.fill([Environment.stateSpace[s,1], Environment.stateSpace[s,1], Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]],
                             [Environment.stateSpace[s,0], Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0], Environment.stateSpace[s,0]], color, alpha = distribution[s])
                    axes.text(Environment.stateSpace[s,1]+0.2, Environment.stateSpace[s,0]+0.5, str(round(distribution[s],1)), color='k', fontsize = 7)
                    
                      
                # axes.gca().set_aspect('equal', adjustable='box')
                axes.title.set_text(f"high level option {l}")    
                axes.axis('off')
                
            elif k == 1 and n_row<2:
                if l == 0:
                    color = 'y'
                if l == 1:
                    color = 'r'
                    
                mapsize = Environment.map.shape
                #count walls
                nwalls=0;
                walls = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if Environment.map[i,j]==Environment.WALL:
                            walls = np.append(walls, [[j, i]], 0)
                            nwalls += 1
                            
                axes.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
        
                for i in range(0,nwalls):
                    axes.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
            
                for i in range(0,nwalls):
                    axes.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                    
                distribution = pi_b[:,l]    
                for s in range(0,len(Environment.stateSpace)):        
                    axes.fill([Environment.stateSpace[s,1], Environment.stateSpace[s,1], Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]],
                             [Environment.stateSpace[s,0], Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0], Environment.stateSpace[s,0]], color, alpha = distribution[s])
                    axes.text(Environment.stateSpace[s,1]+0.2, Environment.stateSpace[s,0]+0.5, str(round(distribution[s],1)), color='k', fontsize = 7)
                    
                      
                # axes.gca().set_aspect('equal', adjustable='box')
                axes.title.set_text(f"termination condition option {l}")    
                axes.axis('off')
                
            elif k == 1 and n_row>=2:
                if l == 0:
                    color = 'y'
                if l == 1:
                    color = 'r'
                    
                mapsize = Environment.map.shape
                #count walls
                nwalls=0;
                walls = np.empty((0,2),int)
                for i in range(0,mapsize[0]):
                    for j in range(0,mapsize[1]):
                        if Environment.map[i,j]==Environment.WALL:
                            walls = np.append(walls, [[j, i]], 0)
                            nwalls += 1
                            
                axes.plot([0, mapsize[1], mapsize[1], 0, 0],[0, 0, mapsize[0], mapsize[0], 0],'k-')
        
                for i in range(0,nwalls):
                    axes.plot([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k-')
            
                for i in range(0,nwalls):
                    axes.fill([walls[i,0], walls[i,0], walls[i,0]+1, walls[i,0]+1, walls[i,0]],
                             [walls[i,1], walls[i,1]+1, walls[i,1]+1, walls[i,1], walls[i,1]],'k')
                    
                distribution = values_state_option[:,l]/np.max(values_state_option)    
                for s in range(0,len(Environment.stateSpace)):        
                    axes.fill([Environment.stateSpace[s,1], Environment.stateSpace[s,1], Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]+1, Environment.stateSpace[s,1]],
                             [Environment.stateSpace[s,0], Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0]+1, Environment.stateSpace[s,0], Environment.stateSpace[s,0]], color, alpha = distribution[s])
                    axes.text(Environment.stateSpace[s,1]+0.2, Environment.stateSpace[s,0]+0.5, str(round(distribution[s],1)), color='k', fontsize = 7)
                    
                      
                # axes.gca().set_aspect('equal', adjustable='box')
                axes.title.set_text(f"state value function option {l}")    
                axes.axis('off')        
            
    
    if not os.path.exists(f"./Figures/{name_env}"):
        os.makedirs(f"./Figures/{name_env}")
    
    plt.savefig(f'Figures/{name_env}/' + name_env + '_summary.pdf', format='pdf')    
    # plt.savefig('HIL_ablation_real_Trajs.jpg', format='jpg')    
    plt.show()