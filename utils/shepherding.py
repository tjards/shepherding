#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 20:00:12 2023

@author: tjards

Refs:
    https://royalsocietypublishing.org/doi/10.1098/rsos.230015


Dev notes:
    
 successfully editing cmds. Herd cmds seens done. Work on cmds shepherds.


"""

# Note: investigate "heterogeneous control strategies"
# - loan wolf, actually goes around the other side to catch/trap the herd
# - how to decide? what criteria? Maybe, if the network gets too big


#%% import stuff
# ------------
import numpy as np
from scipy.spatial.distance import cdist
import copy

#%% hyperparameters
# -----------------
nShepherds = 3  # number of shepherds (just herding = 0)

# for herding
r_R = 2         # repulsion radius
r_O = 3         # orientation radius
r_A = 4         # attraction radius (r_R < r_O < r_A)
r_I = 5.5       # agent interaction radius (nominally, slighly < r_A)

a_R = 2         # gain,repulsion 
a_O = 1         # gain orientation 
a_A = 1         # gain, attraction 
a_I = 4         # gain, agent interaction 
a_V = 4         # gain, laziness (desire to stop)

# for shepherding 
r_S     = r_I - 1           # desired radius from herd
r_Oi    = 3                 # range to view obstacles (here, nearest shepherd)
r_Od    = 1                 # desired distance from obtacles 
r_Or    = 0.5               # radius of shepherd (uniform for all agents, for now)

a_N     = 5                 # gain, navigation
a_R_s   = 1                 # gain, shepards repel eachother
a_R_s_v = 1*np.sqrt(a_R_s)  # gain, shepherds repel eachther (velo component)
a_V_s   = 1*np.sqrt(a_N)    # gain, laziness (desire to stop)

# type of shepherding 
type_shepherd = 'haver'
    #   'haver         = traditional approach to shepherding

# type of collision avoidance for shepherds
type_avoid = 'ref_point'
    #   'ref_shepherd' = maintains rO_d from nearest shepherd
    #   'ref_point'    = (prefered) maintains rO_d from desired location between herd and inv-target 

# use heterogeneous strategies for capturing?
capture     = 1         # 0 = no, 1 = yes
r_c         = r_Oi      # range at which to consider breaking from neighbours
nNeighbours = 2         # criteria to break out (n number nearby)

# bias unique to each
k_noise = 0.1
noise   = np.random.uniform(-1, 1, (nShepherds,3))



# define overall class
# -----------------
class Shepherding:
    
    def __init__(self, state):
        
        # how many are shepherds
        # ----------------------
        self.nShepherds = nShepherds
        self.nHerd      = state.shape[1] - nShepherds
        
        # states of all
        # -------------
        self.state    = state
        #self.states_q = state[0:3,:]
        #self.states_p = state[3:6,:]
        #self.state_shep_i = np.zeros((self.state.shape[0],self.nShepherds))
        #self.state_herd_i = np.zeros((self.state.shape[0],self.state.shape[1]-self.nShepherds))
        
        # discern shepherds from herd
        # ---------------------------
        #self.distinguish = build_index(nShepherds, states_q)
        self.build_index()
        #self.distinguish()
        
        # instantiate the herd and shepherds
        # ----------------------------------
        self.herd       = self.Herd(self)
        self.shepherds  = self.Shepherds(self) 
        
        # store the targets
        # -----------------
        #self.targets = Targets.targets[0:3,:]
        
        # compute distances between all
        # -----------------------------
        #self.seps_all = compute_seps(states_q)
        self.compute_seps()
        
        # agent currently being explored
        self.i = 0
        # neighbour currently being explored
        self.j = 0
        # cmd adjustment (based on sample time, later, import this)
        self.cmd_adjust = 0.02
        self.cmd = np.zeros((1,3))
        
        # store hyper params
        # ------------------
        self.type_shepherd = type_shepherd
        self.type_avoid = type_avoid 
        
        
    # # separate the shepherds from the herd (not used)
    # # -----------------------------------
    # def distinguish(self):
        
    #     # initiate
    #     # --------
    #     #self.state_shep_i = np.zeros((self.state_i.shape[0],self.nShepherds))
    #     i_s = 0
    #     #self.state_herd_i = np.zeros((self.state_i.shape[0],self.state_i.shape[1]-self.nShepherds))
    #     i_h = 0
        
    #     # distinguish between shepherds and herd
    #     # -------------------------------------
    #     for i in range(0,self.state.shape[1]):
            
    #         # shepherds
    #         if self.index[i] == 1:
    #             self.state_shep_i[:,i_s] = self.state[:,i]
    #             i_s += 1
    #         # herd
    #         else:
    #             self.state_herd_i[:,i_h] = self.state[:,i]
    #             i_h += 1      
    
    
    # define separation
    # ------------------ 
    def compute_seps(self):
        
        #states_q = np.concatenate((self.herd.state, self.shepherds.state), axis = 1)
        #states_q = self.state
        
        self.seps_all = np.zeros((self.state.shape[1],self.state.shape[1]))
        i = 0
        while (i<self.state.shape[1]):
            #seps_all[i:state.shape[1],i]=cdist(state[0:3,i].reshape(1,3), state[0:3,i:state.shape[1]].transpose())
            self.seps_all[i:self.state.shape[1],i]=cdist(self.state[0:3,i].reshape(1,3), self.state[0:3,i:self.state.shape[1]].transpose())
          
            i+=1
        
        self.seps_all = self.seps_all + self.seps_all.transpose()
            
        #return seps_all
    
    # build an index distinguishing shepards from herd (1 = s, 0 = h)
    # --------------------------------------------------------------
    def build_index(self):

        # check to ensure herd is big enough
        # ---------------------------------
        if self.nShepherds > (self.state.shape[1]-1):
            raise ValueError("there needs to be at least one member in the herd ")
            
        # compute size of herd
        # --------------------
        #nHerd = state.shape[1] - nShepherds
        
        # random, for now (later, based on conditions)
        # ---------------
        self.index = list(np.concatenate((np.ones(self.nShepherds, dtype=int), np.zeros(self.nHerd, dtype=int))))
        # Shuffle to distribute 1's and 0's randomly
        #np.random.shuffle(index)
        
        
        #return list(index)
    
    # compute commands (called from outside)
    # ----------------
    def compute_cmd(self, Targets, i):
        
        # store the agent being examined
        self.i = i
        
        # store the targets
        self.targets = Targets.targets[0:3,:]
        
        # compute the separations
        self.compute_seps()
        
        # compute command iaw its membership
        if self.index[self.i] == 0:
            
            # compute the command
            self.herd.compute_cmd(self)
        
        elif self.index[self.i] == 1:
            
            # compute the command
            self.shepherds.compute_cmd(self)
        
        self.cmd = 0.02*self.cmd
            
            
    
    
    # define the herd
    # ---------------        
    #class Herd(HerdAndShepherds):
    class Herd():
    
        def __init__(self, outer):
            
            # radial parameters
            self.r_R = r_R         # repulsion radius
            self.r_O = r_O         # orientation radius
            self.r_A = r_A         # attraction radius (r_R < r_O < r_A)
            self.r_I = r_I         # agent interaction radius (nominally, slighly < r_A)
    
            # gain parameters 
            self.a_R = a_R         # gain,repulsion 
            self.a_O = a_O         # gain orientation 
            self.a_A = a_A         # gain, attraction 
            self.a_I = a_I         # gain, agent interaction 
            self.a_V = a_V         # gain, laziness (desire to stop)
            
            #self.state = outer.state_herd_i
     
            # initialization of the superclass (if needed)
            #HerdAndShepherds.__init__(self, HerdAndShepherds.states_q, HerdAndShepherds.nShepherds)
    
            self.indices = [k for k, m in enumerate(outer.index) if m == 0]
    
        # compute herd commands 
        def compute_cmd(self, outer):
            
            # initialize
            # -----------
            #seps_all = compute_seps(states_q)
            #motion_vector = np.zeros((3,states_q.shape[1]))
            outer.cmd = np.zeros((1,3))
            
            # search through each member of the herd
            outer.j = 0
            
            # search through the all agents
            while (outer.j < (outer.nHerd + outer.nShepherds)):
                
                # if not itself
                if outer.seps_all[outer.i,outer.j] > 0.000001:
                    
                    # urge to stop moving
                    outer.cmd += self.a_V * (-outer.state[3:6,outer.i])                   
                    
                    # if it's a member of the herd
                    if outer.index[outer.j] == 0:
                    
                        # repulsion
                        if outer.seps_all[outer.i,outer.j] < self.r_R:
                            outer.cmd -= self.a_R * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                               
                        # orientation
                        if outer.seps_all[outer.i,outer.j] < self.r_O:
                            outer.cmd += self.a_O * np.divide(outer.state[3:6,outer.j]-outer.state[3:6,outer.i],np.linalg.norm(outer.state[3:6,outer.j]-outer.state[3:6,outer.i]))
                        
                        # attraction
                        if outer.seps_all[outer.i,outer.j] < self.r_A:
                            outer.cmd += self.a_A * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                    
                    # if it is a shepherd        
                    elif outer.index[outer.j] == 1:
                    
                        # shepherd influence
                        if outer.seps_all[outer.i,outer.j] < self.r_I:
                            outer.cmd -= self.a_I * np.divide(outer.state[0:3,outer.j]-outer.state[0:3,outer.i],outer.seps_all[outer.i,outer.j])
                            
                outer.j+=1
            
    

    # define the shepherds
    # --------------------
    class Shepherds():
        
        def __init__(self, outer):
            
            # radial parameters
            self.r_S     = r_S      # desired radius from herd
            self.r_Oi    = r_Oi     # range to view obstacles (here, nearest shepherd)
            self.r_Od    = r_Od     # desired distance from obtacles 
            self.r_Or    = r_Or     # radius of shepherd (uniform for all agents, for now)
    
            # gain parameters
            self.a_N     = a_N      # gain, navigation
            self.a_R_s   = a_R_s    # gain, shepards repel eachother
            self.a_R_s_v = a_R_s_v  # gain, shepherds repel eachther (velo component)
            self.a_V_s   = a_V_s    # gain, laziness (desire to stop)

            #self.state = outer.state_shep_i
            
            self.indices = [k for k, m in enumerate(outer.index) if m == 1]

        # compute herd commands 
        def compute_cmd(self, outer):
                
            outer.cmd = np.zeros((3,1))

            seps_list       = list(outer.seps_all[outer.i,:])
            
            
            # make all the shepherds negative
            for k in self.indices:
                seps_list[k] = -seps_list[k]

            #closest_herd    = seps_list.index(min(k for k in seps_list if k in self.indices.remove(outer.i)))
            #closest_herd    = seps_list.index(min(k for k in seps_list if k > 0))               
            

            # find the indices for the shepherds
            #indices_shep = [k for k, m in enumerate(distinguish) if m == 1]
            
            # make them negative
            #for k in indices_shep:
            #    seps_list[k] = -seps_list[k]
             
            # find the closest herd
            closest_herd        = seps_list.index(min(k for k in seps_list if k > 0))
    

            
            
            # compute the normalized vector between closest in herd and target 
            #v = np.divide(states_q[:,closest_herd]-targets[:,i],np.linalg.norm(states_q[:,closest_herd]-targets[:,i])) 
            
            v = np.divide(outer.state[0:3,closest_herd]-outer.targets[0:3,outer.i],np.linalg.norm(outer.state[0:3,closest_herd]-outer.targets[0:3,outer.i])) 
     
            
            # compute the desired location to shepard (based on closets hearder)
            q_s = outer.state[0:3,closest_herd] + self.r_S*v  # location
            d_s = np.linalg.norm(q_s-outer.state[0:3,outer.i]) # distance
            
            
            # find the closest shepherd
            closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
            
            q_cs = outer.state[0:3,closest_shepherd]         # location of closest shepherd
            d_cs = np.linalg.norm(q_cs-outer.state[0:3,outer.i])   # distance from that closest shepherd
            
            
            # if using havermaet technique
            # ----------------------------
            if outer.type_shepherd == 'haver':
            
                # navigate to push the herd towards targets
                # -----------------------------------------
                outer.cmd = self.a_N * np.divide(q_s-outer.state[0:3,outer.i],np.linalg.norm(q_s-outer.state[0:3,outer.i]))
                
                # urge to slow down
                # ----------------
                outer.cmd += self.a_V_s * (-outer.state[3:6,outer.i])
                
                # if the closet shepherd is within avoidance range
                if d_cs < self.r_Oi:
                    
                    # avoid the shepherd
                    if outer.type_avoid == 'ref_shepherd':
                    
                        bold_a_k = np.array(np.divide(outer.state[0:3,outer.i]-q_cs,d_cs), ndmin = 2)
                        P = np.identity(outer.state[3:6,:].shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                        mu = np.divide(self.r_Or,d_cs) 
                        p_ik = mu*np.dot(P,outer.state[3:6,outer.i]) 
                        q_ik = mu*outer.state[0:3,outer.i]+(1-mu)*q_cs
                                    
                        outer.cmd += self.a_R_s*phi_b(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*n_ij(outer.state[0:3,outer.i], q_ik) + self.a_R_s_v*b_ik(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*(p_ik - outer.state[3:6,outer.i])
                 
                    # avoid the reference point (ends up working nicely)
                    elif outer.type_avoid == 'ref_point':
                        
                        bold_a_k = np.array(np.divide(outer.state[0:3,outer.i]-q_s,d_s), ndmin = 2)
                        P = np.identity(outer.state[3:6,:].shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                        mu = np.divide(self.r_Or,d_s) 
                        p_ik = mu*np.dot(P,outer.state[3:6,outer.i]) 
                        q_ik = mu*outer.state[0:3,outer.i]+(1-mu)*q_s
                                    
                        outer.cmd += self.a_R_s*phi_b(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*n_ij(outer.state[0:3,outer.i], q_ik) + self.a_R_s_v*b_ik(outer.state[0:3,outer.i], q_ik, sigma_norm(self.r_Od))*(p_ik - outer.state[3:6,outer.i])
          
                
                


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



# separate the shepherds from the herd (not used)
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

# this is for finding n closest shepherds
# ---------------------------------------
def find_n_neighbours(n, sepslist):
    
    neighbours = []
    
    for _ in range(0,n):
        
        # find the min value 
        select = sepslist.index(max(k for k in sepslist if k < 0))
        
        # add to list 
        neighbours.append(select)
        
        # exclude this index for next round
        sepslist[select] = float('inf')
        
    return neighbours
       
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
     
        
     
    # find the closest herd
    closest_herd        = seps_list.index(min(k for k in seps_list if k > 0))

    
    
    # compute the normalized vector between closest in herd and target 
    v = np.divide(states_q[:,closest_herd]-targets[:,i],np.linalg.norm(states_q[:,closest_herd]-targets[:,i])) 
     
    
    # compute the desired location to shepard (based on closets hearder)
    q_s = states_q[:,closest_herd] + r_S*v  # location
    d_s = np.linalg.norm(q_s-states_q[:,i]) # distance
    
    # find the closest shepherd
    closest_shepherd    = seps_list.index(max(k for k in seps_list if k < 0))
    
    q_cs = states_q[:,closest_shepherd]         # location of closest shepherd
    d_cs = np.linalg.norm(q_cs-states_q[:,i])   # distance from that closest shepherd
    
    

    # if using capturing, check if criteria met
    # if capture == 1:
        
    #     # find n neighbours
    #     neighbours              = find_n_neighbours(nNeighbours, copy.deepcopy(seps_list))
    #     neighbours_distances    = [seps_list[i] for i in neighbours]
        
    #     # if they are all close enough
    #     if all(k > -r_c for k in neighbours_distances):
    #         print('note: need to objectify this module')
    #         print('because I want this agent to break away')

       
    # if using havermaet technique
    # ----------------------------
    if type_shepherd == 'haver':
    
        # navigate to push the herd towards targets
        # -----------------------------------------
        cmd = a_N * np.divide(q_s-states_q[:,i],np.linalg.norm(q_s-states_q[:,i]))
        
        # urge to slow down
        # ----------------
        cmd += a_V_s * (-states_p[:,i])
        
        # if the closet shepherd is within avoidance range
        if d_cs < r_Oi:
            
            # avoid the shepherd
            if type_avoid == 'ref_shepherd':
            
                bold_a_k = np.array(np.divide(states_q[:,i]-q_cs,d_cs), ndmin = 2)
                P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                mu = np.divide(r_Or,d_cs) 
                p_ik = mu*np.dot(P,states_p[:,i]) 
                q_ik = mu*states_q[:,i]+(1-mu)*q_cs
                            
                cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])
         
            # avoid the reference point (ends up working nicely)
            elif type_avoid == 'ref_point':
                
                bold_a_k = np.array(np.divide(states_q[:,i]-q_s,d_s), ndmin = 2)
                P = np.identity(states_p.shape[0]) - np.multiply(bold_a_k,bold_a_k.transpose())
                mu = np.divide(r_Or,d_s) 
                p_ik = mu*np.dot(P,states_p[:,i]) 
                q_ik = mu*states_q[:,i]+(1-mu)*q_s
                            
                cmd += a_R_s*phi_b(states_q[:,i], q_ik, sigma_norm(r_Od))*n_ij(states_q[:,i], q_ik) + a_R_s_v*b_ik(states_q[:,i], q_ik, sigma_norm(r_Od))*(p_ik - states_p[:,i])
  
    return cmd + noise[i,:]
  

# old one
# -------  
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
    

    
    
#%%
# # test

# nVeh = 7

# class test_target:
    
#     def __init__(self):
        
#         self.targets = 4*(np.random.rand(6,nVeh)-0.5)
#         self.targets[0,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
#         self.targets[1,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
#         self.targets[2,:] = 15
#         self.targets[3,:] = 0
#         self.targets[4,:] = 0
#         self.targets[5,:] = 0
    
    


# Targets = test_target()
# new = Shepherding(np.zeros((6,nVeh)),3, Targets)
# #import random
# #rows = 6
# #cols = 7
# # Create a 6x7 array with random float values
# #new.state = np.array([[random.random() for _ in range(cols)] for _ in range(rows)])

# new.compute_cmd(6)
# cmd = new.cmd 
# pin_matrix = new.index[new.i]

#%% LEGACY code
# -------------

# # store data
# # ----------
# data = {}
# data['type_shepherd'] = type_shepherd
# data['r_R']         = r_R
# data['r_O']         = r_O
# data['r_A']         = r_A
# data['r_I']         = r_I
# data['a_R']         = a_R
# data['a_O']         = a_O  
# data['a_A']         = a_A
# data['a_I']         = a_I
# data['a_V']         = a_V
# data['r_S']         = r_S   
# data['a_N']         = a_N    
# data['a_R_s']       = a_R_s 
# data['a_R_s_v']     = a_R_s_v
# data['a_V_s']       = a_V_s  
# data['r_Oi']        = r_Oi 
# data['r_Od']        = r_Od 
# data['r_Or']        = r_Or   

# current_datetime = datetime.now()
# formatted_date = current_datetime.strftime("%Y%m%d_%H%M%S")
# #current_directory = os.getcwd()
# #parent_directory = os.path.dirname(current_directory)
# #data_directory = os.path.join(parent_directory, 'Data')
# data_directory = 'Data'
# file_path = os.path.join(data_directory, f"data_params_shepherding_{formatted_date}.json")
    
# with open(file_path, 'w') as file:
#     json.dump(data, file)


              
    # elif type_shepherd == 'pin_net':
        
        
    #     #cmd_i[:,i] = pinning_tools.compute_cmd(centroid, states_q, states_p, obstacles, walls, targets, targets_v, i, pin_matrix)
    #     # some messy substitutions just to get this working (need to add obs/walls later):
        
    #     # i only want to deal with the shepherds
    #     #columns_to_delete = np.where(np.array(distinguish) == 0)[0]
    #     shep_q = np.delete(states_q, np.where(np.array(distinguish) == 0)[0], axis=1)
    #     shep_p = np.delete(states_p, np.where(np.array(distinguish) == 0)[0], axis=1)
            
    #     walls_temp = np.array([[-0.5 ],[ 0.25],[50.  ],[ 0.  ],[ 0.  ],[-2.  ]])
    #     targets_shep = np.zeros((3,shep_q.shape[1]))
    #     targets_shep_v = np.zeros((3,shep_p.shape[1]))
    #     targets_shep[0:3,:] = q_s.reshape(3,1)
    #     #targets_shep_v[0:3,:] = pc_s.reshape(3,1)
        
    #     # inefficient: this should be passed in      
    #     pin_matrix_shep, components = pinning_tools.select_pins_components(shep_q) 
        
    #     # clean this up
    #     cmd = pinning_tools.compute_cmd(centroid, shep_q, shep_p, np.zeros((4,1)), walls_temp, targets_shep, targets_shep_v, i, pin_matrix_shep)     
            

    
    # # messy, but I want to show the shepherding pins as a different color, so    
    # if type_shepherd == 'pin_net':
    #     pin_matrix, _ = pinning_tools.select_pins_components(states_q)
    #     if pin_matrix[i,i] == 1 and distinguish[i] == 1:
    #         distinguish[i] = 2
