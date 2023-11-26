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
from utils import quaternions as quat

#%% hyperparameters
# -----------------
nShepherds = 3  # number of shepherds (just herding = 0)

# for herding
r_R = 2         # repulsion radius
r_O = 4         # orientation radius
r_A = 7         # attraction radius (a_R < a_O < a_A)
r_I = 6.5       # agent interaction radius (nominally, slighly < a_A)

a_R = 0.5       # gain,repulsion 
a_O = 1         # gain orientation 
a_A = 0.8       # gain, attraction 
a_I = 1.2        # gain, agent interaction 
a_V = 0.2      # gain, laziness (desire to stop)

# for shepherding
r_S     = r_I - 1           # desired radius from herd
a_N     = 5                 # gain, navigation
a_R_s   = 1                 # gain, shepards repel eachother
a_R_s_v = 1*np.sqrt(a_R_s)
a_V_s   = 1*np.sqrt(a_N)    # gain, laziness (desire to stop)

r_Oi    = 2                 # range to view obstacles (here, nearest shepherd)
r_Od    = 1                 # desired distance from obtacles 
r_Or    = 0.5               # radius of shepherd (uniform for all agents, for now)

# techniques 
shepherd_type = 'haver'




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

# define separation
# ------------------ 
def compute_seps(state):
    seps_all = np.zeros((state.shape[1],state.shape[1]))
    i = 0
    while (i<state.shape[1]):
        seps_all[i:state.shape[1],i]=cdist(state[0:3,i].reshape(1,3), state[0:3,i:state.shape[1]].transpose())
        i+=1
    
    seps_all = seps_all + seps_all.transpose()
        
    return seps_all

# obstacle avoidance stuff (shepherds)
# ------------------------------------
eps = 0.1
h   = 0.9
pi  = 3.141592653589793

def sigma_1(z):    
    sigma_1 = np.divide(z,np.sqrt(1+z**2))    
    return sigma_1

def rho_h(z):    
    if 0 <= z < h:
        rho_h = 1        
    elif h <= z < 1:
        rho_h = 0.5*(1+np.cos(pi*np.divide(z-h,1-h)))    
    else:
        rho_h = 0  
    return rho_h

def sigma_norm(z):    
    norm_sig = (1/eps)*(np.sqrt(1+eps*np.linalg.norm(z)**2)-1)
    return norm_sig

def phi_b(q_i, q_ik, d_b): 
    z = sigma_norm(q_ik-q_i)        
    phi_b = rho_h(z/d_b) * (sigma_1(z-d_b)-1)    
    return phi_b

def n_ij(q_i, q_j):
    n_ij = np.divide(q_j-q_i,np.sqrt(1+eps*np.linalg.norm(q_j-q_i)**2))    
    return n_ij

def b_ik(q_i, q_ik, d_b):        
    b_ik = rho_h(sigma_norm(q_ik-q_i)/d_b)
    return b_ik

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
    
    # initialize
    cmd = np.zeros((3,states_q.shape[1]))
    
    # find the indices for the shepherds
    indices_shep = [k for k, m in enumerate(distinguish) if m == 1]
    
    # make them negative
    for k in indices_shep:
        seps_list[k] = -seps_list[k]
        
    # if using havermaet technique
    # ----------------------------
    if shepherd_type == 'haver':
    
        # deal with the herd
        # ------------------
        
        # find the closest herd
        closest_herd        = seps_list.index(min(k for k in seps_list if k > 0))
        
        # compute the normalized vector between closest in herd and target 
        v = np.divide(states_q[:,closest_herd]-targets[:,i],np.linalg.norm(states_q[:,closest_herd]-targets[:,i])) 
        
        # compute the desired location to shepard
        q_s = states_q[:,closest_herd] + r_S*v 
        
        # navigate to that position
        cmd = a_N * np.divide(q_s-states_q[:,i],np.linalg.norm(q_s-states_q[:,i]))
        
        # urge to slow down
        # ----------------
        cmd += a_V_s * (-states_p[:,i])
        
        # deal with other shepherds (try just the angular separation piece of encirclement?)
        # -------------------------
        type_avoid = 'ref_point'
        #    'ref_shepherd' = maintains rO_d from nearest shepherd
        #    'ref_point'    = maintains rO_d from desired location between herd and inv-target       
        
        closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
        q_cs = states_q[:,closest_shepherd]         # closest shepherd
        d_cs = np.linalg.norm(q_cs-states_q[:,i])   # distance from that closest shepherd
        
        # maintain a desired separation from closest shepard (if within range)
        if d_cs < r_Oi:
            
            if type_avoid == 'ref_shepherd':
            
                bold_a_k = np.array(np.divide(states_q[:,i]-q_cs,d_cs), ndmin = 2)
                P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                mu = np.divide(r_Or,d_cs) 
                p_ik = mu*np.dot(P,states_p[:,i]) 
                q_ik = mu*states_q[:,i]+(1-mu)*q_cs
                            
                cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])
         
            elif type_avoid == 'ref_point':
                
                d_s = np.linalg.norm(q_s-states_q[:,i])
                
                bold_a_k = np.array(np.divide(states_q[:,i]-q_s,d_s), ndmin = 2)
                P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                mu = np.divide(r_Or,d_s) 
                p_ik = mu*np.dot(P,states_p[:,i]) 
                q_ik = mu*states_q[:,i]+(1-mu)*q_s
                            
                cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])

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
    

