#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:00:12 2023

@author: tjards

Refs:
    https://royalsocietypublishing.org/doi/10.1098/rsos.230015


"""

# Note: doesn't allow shepards < 2 (because of min/max stuff), fix this later


#%% import stuff
# ------------
import numpy as np
from scipy.spatial.distance import cdist

#%% hyperparameters
# -----------------
nShepherds = 10  # number of shepherds (just herding = 0)

# herd
r_R = 2         # repulsion radius
r_O = 4         # orientation radius
r_A = 7         # attraction radius (a_R < a_O < a_A)
r_I = 6.5       # agent interaction radius (nominally, slighly < a_A)

a_R = 0.5       # gain,repulsion 
a_O = 4         # gain orientation 
a_A = 2         # gain, attraction 
a_I = 2         # gain, agent interaction 
a_V = 0.25      # gain, laziness (desire to stop)

# shepherds
r_S = r_I - 1   # desired radius from herd

a_N = 5        # gain, navigation
a_R_s = 10       # gain, shepards repel eachother
a_V_s = np.sqrt(2)*2      # gain, laziness (desire to stop)

# build an index distinguishing shepards from herd (1 = s, 0 = h)
# --------------------------------------------------------------
def build_index(nShepherds, state):

    # check to ensure herd is big enough
    # ---------------------------------
    if nShepherds > (state.shape[1]-1):
        raise ValueError("there needs to be at least one member in the herd ")
        
    # compute size of herd
    # --------------------
    nHerd = state.shape[1] - nShepherds
    
    # random, for now (later, based on conditions)
    # ---------------
    index = np.concatenate((np.ones(nShepherds, dtype=int), np.zeros(nHerd, dtype=int)))
    # Shuffle to distribute 1's and 0's randomly
    #np.random.shuffle(index)
    
    return list(index)

# separate the shepherds from the herd
# -----------------------------------
def distinguish(state, nShepherds, index):
    
    # initiate
    # --------
    shepherds = np.zeros((state.shape[0],nShepherds))
    i_s = 0
    herd = np.zeros((state.shape[0],state.shape[1]-nShepherds))
    i_h = 0
    
    # distinguish between shepherds and herd
    # -------------------------------------
    for i in range(0,state.shape[1]):
        
        # shepherds
        if index[i] == 1:
            shepherds[:,i_s] = state[:,i]
            i_s += 1
        # herd
        else:
            herd[:,i_h] = state[:,i]
            i_h += 1    
    
    return shepherds, herd

#%% define separation
# ------------------ 
def compute_seps(state):
    seps_all = np.zeros((state.shape[1],state.shape[1]))
    i = 0
    while (i<state.shape[1]):
        seps_all[i:state.shape[1],i]=cdist(state[0:3,i].reshape(1,3), state[0:3,i:state.shape[1]].transpose())
        i+=1
    
    seps_all = seps_all + seps_all.transpose()
        
    return seps_all
               
# compute command - herd
# ----------------------------
def compute_cmd_herd(states_q, states_p, i, distinguish, seps_all):
    
    # initialize
    # -----------
    #seps_all = compute_seps(states_q)
    motion_vector = np.zeros((3,states_q.shape[1]))
    
    # search through each agent
    j = 0
    while (j < states_q.shape[1]):
        
        # but not itself
        if i != j:
            
            # pull distance
            dist = seps_all[i,j]
            #print(dist)
            
            # I could nest these, given certain radial constraints
            # ... but I won't, deliberately, for now (enforce above, then come back later)
            #print(i)
            
            # urge to stop moving
            motion_vector[:,i] += a_V * (-states_p[:,i])
              
            # repulsion
            if dist < r_R and distinguish[j] == 0:
                motion_vector[:,i] -= a_R * np.divide(states_q[:,j]-states_q[:,i],dist)
                   
            # orientation
            if dist < r_O and distinguish[j] == 0:
                motion_vector[:,i] += a_O * np.divide(states_p[:,j]-states_p[:,i],np.linalg.norm(states_p[:,j]-states_p[:,i]))
            
            # attraction
            if dist < r_A and distinguish[j] == 0:
                motion_vector[:,i] += a_A * np.divide(states_q[:,j]-states_q[:,i],dist)
                
            # shepherd influence
            if dist < r_I and distinguish[j] == 1:
                motion_vector[:,i] -= a_I * np.divide(states_q[:,j]-states_q[:,i],dist)
                
        j+=1
    
    return motion_vector[:,i] 

# compute commands - sheperd
# -------------------------
def compute_cmd_shep(targets, centroid, states_q, states_p, i, distinguish, seps_list):
    
    # find the indices for the shepherds
    indices_shep = [k for k, m in enumerate(distinguish) if m == 1]
    # make them negative
    for k in indices_shep:
        seps_list[k] = -seps_list[k]
    # find the closest shepherd
    closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
    # find the closest herd
    closest_herd        = seps_list.index(min(k for k in seps_list if k > 0))
    
    # push the center of the herd
    # ---------------------------
    # compute the normalized vector between center of herd and target <- this should be closest herd member, not centroid
    #v = np.divide(centroid.transpose()-targets[:,i],np.linalg.norm(centroid.transpose()-targets[:,i])) 
    
    # compute the normalized vector between closest in herd and target 
    v = np.divide(states_q[:,closest_herd]-targets[:,i],np.linalg.norm(states_q[:,closest_herd]-targets[:,i])) 
    
    # compute the desired location to shepard
    #q_s = centroid.transpose() + r_S*v  
    q_s = states_q[:,closest_herd] + r_S*v # travis, I think this needs to be +
    
    # navigate to that position
    cmd = a_N * np.divide(q_s-states_q[:,i],np.linalg.norm(q_s-states_q[:,i]))
    
    # slowdown
    # --------
    cmd += a_V_s * (-states_p[:,i])
    
    # repel from closest shepherd
    # ---------------------------
    q_h = states_q[:,closest_shepherd]
    cmd += a_R_s * np.divide(q_h-states_q[:,i],np.linalg.norm(q_h-states_q[:,i]))
    
    
    # # repel from other shepherd (legacy)
    # # -------------------------
    # j = 0
    # # go thru agents
    # while (j < states_q.shape[1]):
    #     # if its another shepherd
    #     if i != j and distinguish[j] == 1:
    #         # compute a repulsive command (just away)
    #         cmd -= a_R_s * np.divide(states_q[:,j]-states_q[:,i],np.linalg.norm(states_q[:,j]-states_q[:,i]))
    #     j += 1
    
    return cmd
    

def compute_cmd(targets, centroid, states_q, states_p, i):
    
    
    # compute distances between all
    # -----------------------------
    seps_all = compute_seps(states_q)
    
    # discern shepherds from herd
    # ---------------------------
    distinguish = build_index(nShepherds, states_q)
    
    # if it is a member of the herd
    # ----------------------------
    if distinguish[i] == 0:
    
        # do the herd stuff
        # -----------------
        cmd = compute_cmd_herd(states_q, states_p, i, distinguish, seps_all)
    
    else:
        
        # do the shepherd stuff
        # ----------------------
        cmd =  compute_cmd_shep(targets,centroid, states_q, states_p, i, distinguish, list(seps_all[i,:]))   
        
    return cmd*0.02, distinguish[i] #note, this is Ts, because output of above is velo, model is double integrator
    


#%% DEV
    
# run
# ----
#index = build_index(nShepherds, state.shape[1]-nShepherds)
# print(index)  
#shepherds, herd = distinguish(state, nShepherds, index)
# print(shepherds)
# print(herd) 
# seps_all = compute_seps(state)   
# print(seps_all)
#print(motion_intraherd(herd))




# define motion vector 
# ----------------------
# def motion_intraherd(state):
    
#     seps_all = compute_seps(state)
#     motion_vector = np.zeros((3,state.shape[1]))
    
#     # search through each agent
#     for i in range(0,state.shape[1]):
#         # and others
#         j = i
#         while (j < state.shape[1]):
#             # but not itself
#             if i != j:
                
#                 # pull distance
#                 dist = seps_all[j,i]
#                 print(dist)
                
#                 # I could nest these, given certain radial constraints
#                 # ... but I won't, deliberately, for now (enforce above, then come back later)
#                 print(i)
#                 # repulsion
#                 if dist < r_R:
#                     motion_vector[:,i] -= a_R * np.divide(state[0:3,j]-state[0:3,i],dist)
                       
#                 # orientation
#                 if dist < r_O:
#                     motion_vector[:,i] += a_O * np.divide(state[3:6,j]-state[3:6,i],np.linalg.norm(state[3:6,j]-state[3:6,i]))
                
#                 # alignment
#                 if dist < r_A:
#                     motion_vector[:,i] += a_A * np.divide(state[0:3,j]-state[0:3,i],dist)

#             j+=1
    
#     return motion_vector 



# state = np.array([[-0.09531187, -3.62677053, -0.19474599,  0.60565189, -0.49085531],
#         [ 0.7286528 ,  0.29258714,  1.55372048, -2.22041092, -3.20395583],
#         [19.55264931, 15.97155896, 12.54037952, 15.66294835, 10.96263862],
#         [-0.04973982, -0.17979188, -0.04870185,  0.02474909, -0.05649268],
#         [ 0.11554307,  0.35100273,  0.12836373,  0.27391689,  0.22342425],
#         [-0.22158579, -0.26174345, -0.04171904, -0.20590812, -0.46992463]])

# targets = np.array([[ 0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.],
#         [15., 15., 15., 15., 15.],
#         [ 0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.],
#         [ 0.,  0.,  0.,  0.,  0.]])

