#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:00:12 2023

@author: tjards


"""




#%% import stuff
# ------------
import numpy as np

# need to break into two parts
# 1. heard
# 2. shepherds

#%% hyperparameters
# -----------------
nShepherds = 1      # number of shepherds (default = 1, just herding = 0)


#%% dev
# -----

state = np.array([[-0.09531187, -3.62677053, -0.19474599,  0.60565189, -0.49085531],
       [ 0.7286528 ,  0.29258714,  1.55372048, -2.22041092, -3.20395583],
       [19.55264931, 15.97155896, 12.54037952, 15.66294835, 10.96263862],
       [-0.04973982, -0.17979188, -0.04870185,  0.02474909, -0.05649268],
       [ 0.11554307,  0.35100273,  0.12836373,  0.27391689,  0.22342425],
       [-0.22158579, -0.26174345, -0.04171904, -0.20590812, -0.46992463]])

targets = np.array([[ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [15., 15., 15., 15., 15.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.]])


# build an index distinguishing shepards from herd (1 = s, 0 = h)
# --------------------------------------------------------------
def build_index(nShepherds, nHerd):
    
    if nShepherds > (state.shape[1]-1):
        raise ValueError("there needs to be at least one member in the herd ")
    # Create an array with m 1's and (n-m) 0's
    index = np.concatenate((np.ones(nShepherds, dtype=int), np.zeros(nHerd, dtype=int)))
    # Shuffle the array to distribute 1's and 0's randomly
    np.random.shuffle(index)
    return index

# separate the shepherds from the herd
# -----------------------------------
def distinguish(state, nShepherds, index):
    
    # initiate
    # --------
    shepherds = np.zeros((state.shape[0],nShepherds))
    i_s = 0
    herd = np.zeros((state.shape[0],state.shape[1]-nShepherds))
    i_h = 0
    
    # fill
    # ----
    for i in range(0,state.shape[1]):
        
        # shepherd
        if index[i] == 1:
            shepherds[:,i_s] = state[:,i]
            i_s += 1
        # herd
        else:
            herd[:,i_h] = state[:,i]
            i_h += 1    
    
    return shepherds, herd

    
#%% run
# ----
index = build_index(nShepherds, state.shape[1]-nShepherds)
print(index)  

shepherds, herd = distinguish(state, nShepherds, index)
print(shepherds)
print(herd)    