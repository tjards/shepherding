#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This project implements an autonomous, decentralized swarming strategies including:
    
    - Reynolds rules of flocking ("boids")
    - Olfati-Saber flocking
    - Starling flocking
    - Dynamic Encirclement 
    - Pinning Control
    - Autonomous Assembly of Closed Curves

The strategies requires no human invervention once the target is selected and all agents rely on local knowledge only. 
Each vehicle makes its own decisions about where to go based on its relative position to other vehicles.

Created on Tue Dec 22 11:48:18 2020

@author: tjards

"""

#%% Import stuff
# --------------

# official packages 
#from scipy.integrate import ode
import numpy as np
import pickle 
import matplotlib.pyplot as plt
#plt.style.use('dark_background')
#plt.style.use('classic')
plt.style.use('default')
#plt.style.available
#plt.style.use('Solarize_Light2')
import copy
import random

# from root folder
#import animation 
import animation
import dynamics_node as node
import ctrl_tactic as tactic 

# utilities 
from utils import encirclement_tools as encircle_tools
from utils import pinning_tools, lemni_tools, starling_tools, swarm_metrics, tools, modeller
#from utils import graph_tools

#%% Setup Simulation
# ------------------
np.random.seed(2)
Ti      =   0       # initial time
Tf      =   15     # final time (later, add a condition to break out when desirable conditions are met)
Ts      =   0.02    # sample time
nVeh    =   5      # number of vehicles
iSpread =   15      # initial spread of vehicles
tSpeed  =   0       # speed of target
rVeh    =   0.5     # physical radius of vehicle 
exclusion = []      # initialization of what agents to exclude, default empty

tactic_type = 'pinning'     
                # reynolds = Reynolds flocking + Olfati-Saber obstacle
                # saber = Olfati-Saber flocking
                # starling = swar like starlings 
                # circle = encirclement
                # lemni = dynamic lemniscates and other closed curves
                # pinning = pinning control

# if using reynolds, need make target an obstacle 
if tactic_type == 'reynolds':
    targetObs = 1
else:
    targetObs = 0    
    
# do we want to build a model in real time?
#real_time_model = 'yes'

# Vehicles states
# ---------------
state = np.zeros((6,nVeh))
state[0,:] = iSpread*(np.random.rand(1,nVeh)-0.5)                   # position (x)
state[1,:] = iSpread*(np.random.rand(1,nVeh)-0.5)                   # position (y)
state[2,:] = np.maximum((iSpread*np.random.rand(1,nVeh)-0.5),2)+15  # position (z)
state[3,:] = 0                                                      # velocity (vx)
state[4,:] = 0                                                      # velocity (vy)
state[5,:] = 0                                                      # velocity (vz)
centroid = tools.centroid(state[0:3,:].transpose())
centroid_v = tools.centroid(state[3:6,:].transpose())
# select a pin (for pinning control)
pin_matrix = np.zeros((nVeh,nVeh))
if tactic_type == 'pinning':
    pin_matrix = pinning_tools.select_pins_components(state[0:3,:])

# Commands
# --------
cmd = np.zeros((3,nVeh))
cmd[0] = np.random.rand(1,nVeh)-0.5      # command (x)
cmd[1] = np.random.rand(1,nVeh)-0.5      # command (y)
cmd[2] = np.random.rand(1,nVeh)-0.5      # command (z)

# Targets
# -------
targets = 4*(np.random.rand(6,nVeh)-0.5)
targets[0,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
targets[1,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
targets[2,:] = 15
targets[3,:] = 0
targets[4,:] = 0
targets[5,:] = 0
targets_encircle = targets.copy()
error = state[0:3,:] - targets[0:3,:]

# Other Parameters
# ----------------
params = np.zeros((4,nVeh))  # store dynamic parameters

# do I want to model in realtime?
#if real_time_model == 'yes':
#    swarm_model = modeller.model()


#%% Define obstacles (kind of a manual process right now)
# ------------------------------------------------------
nObs    = 0     # number of obstacles 
vehObs  = 0     # include other vehicles as obstacles [0 = no, 1 = yes] 

# there are no obstacle, but we need to make target an obstacle 
if nObs == 0 and targetObs == 1:
    nObs = 1

obstacles = np.zeros((4,nObs))
oSpread = 10

# manual (comment out if random)
# obstacles[0,:] = 0    # position (x)
# obstacles[1,:] = 0    # position (y)
# obstacles[2,:] = 0    # position (z)
# obstacles[3,:] = 0

#random (comment this out if manual)
if nObs != 0:
    obstacles[0,:] = oSpread*(np.random.rand(1,nObs)-0.5)+targets[0,0]                   # position (x)
    obstacles[1,:] = oSpread*(np.random.rand(1,nObs)-0.5)+targets[1,0]                   # position (y)
    obstacles[2,:] = oSpread*(np.random.rand(1,nObs)-0.5)+targets[2,0]                  # position (z)
    #obstacles[2,:] = np.maximum(oSpread*(np.random.rand(1,nObs)-0.5),14)     # position (z)
    obstacles[3,:] = np.random.rand(1,nObs)+1                             # radii of obstacle(s)

# manually make the first target an obstacle
if targetObs == 1:
    obstacles[0,0] = targets[0,0]     # position (x)
    obstacles[1,0] = targets[1,0]     # position (y)
    obstacles[2,0] = targets[2,0]     # position (z)
    obstacles[3,0] = 2              # radii of obstacle(s)

# Walls/Floors 
# - these are defined manually as planes
# --------------------------------------   
nWalls = 1                      # default 1, as the ground is an obstacle 
walls = np.zeros((6,nWalls)) 
walls_plots = np.zeros((4,nWalls))

# add the ground at z = 0:
newWall0, newWall_plots0 = tools.buildWall('horizontal', -2) 

# load the ground into constraints   
walls[:,0] = newWall0[:,0]
walls_plots[:,0] = newWall_plots0[:,0]

# add other planes (comment out by default)

# newWall1, newWall_plots1 = flock_tools.buildWall('diagonal1a', 3) 
# newWall2, newWall_plots2 = flock_tools.buildWall('diagonal1b', -3) 
# newWall3, newWall_plots3 = flock_tools.buildWall('diagonal2a', -3) 
# newWall4, newWall_plots4 = flock_tools.buildWall('diagonal2b', 3)

# load other planes (comment out by default)

# walls[:,1] = newWall1[:,0]
# walls_plots[:,1] = newWall_plots1[:,0]
# walls[:,2] = newWall2[:,0]
# walls_plots[:,2] = newWall_plots2[:,0]
# walls[:,3] = newWall3[:,0]
# walls_plots[:,3] = newWall_plots3[:,0]
# walls[:,4] = newWall4[:,0]
# walls_plots[:,4] = newWall_plots4[:,0]

#%% Run Simulation
# ----------------------
t = Ti
i = 1
f = 0         # parameter for future use

nSteps = int(Tf/Ts+1)

# initialize a bunch of storage 
t_all               = np.zeros(nSteps)
states_all          = np.zeros([nSteps, len(state), nVeh])
cmds_all            = np.zeros([nSteps, len(cmd), nVeh])
targets_all         = np.zeros([nSteps, len(targets), nVeh])
obstacles_all       = np.zeros([nSteps, len(obstacles), nObs])
centroid_all        = np.zeros([nSteps, len(centroid), 1])
f_all               = np.ones(nSteps)
lemni_all           = np.zeros([nSteps, nVeh])
# metrics_order_all   = np.zeros((nSteps,7))
# metrics_order       = np.zeros((1,7))
nMetrics            = 12 # there are 11 positions being used.    
metrics_order_all   = np.zeros((nSteps,nMetrics))
metrics_order       = np.zeros((1,nMetrics))
pins_all            = np.zeros([nSteps, nVeh, nVeh]) 
# note: for pinning control, pins denote pins as a 1
# also used in lemni to denote membership in swarm as 0

# store the initial conditions
t_all[0]                = Ti
states_all[0,:,:]       = state
cmds_all[0,:,:]         = cmd
targets_all[0,:,:]      = targets
obstacles_all[0,:,:]    = obstacles
centroid_all[0,:,:]     = centroid
f_all[0]                = f
metrics_order_all[0,:]  = metrics_order
lemni                   = np.zeros([1, nVeh])
lemni_all[0,:]          = lemni
pins_all[0,:,:]         = pin_matrix       

# we need to move the 'target' for mobbing (a type of lemniscate)
if tactic_type == 'lemni':
    targets = lemni_tools.check_targets(targets)
    
#%% start the simulation
# --------------------

while round(t,3) < Tf:
    
    # Evolve the target
    # -----------------
    targets[0,:] = 100*np.sin(tSpeed*t)                 # targets[0,:] + tSpeed*0.002
    targets[1,:] = 100*np.sin(tSpeed*t)*np.cos(tSpeed*t)  # targets[1,:] + tSpeed*0.005
    targets[2,:] = 100*np.sin(tSpeed*t)*np.sin(tSpeed*t)+15  # targets[2,:] + tSpeed*0.0005
    
    # For pinning application, we may set the first agent as the "pin",
    # which means all other targets have to be set to the pin
    # comment out for non-pinning control
    # ------------------------------------------------------------
    #targets[0,1::] = state[0,0]
    #targets[1,1::] = state[1,0]
    #targets[2,1::] = state[2,0]
    
    # Update the obstacles (if required)
    # ----------------------------------
    if targetObs == 1:
        obstacles[0,0] = targets[0,0]     # position (x)
        obstacles[1,0] = targets[1,0]     # position (y)
        obstacles[2,0] = targets[2,0]     # position (z)

    # modeller: load the current states (x,v), centroid states (x,v) and inputs (of the first agent)
    # -------------------------------------------------------------------------------
    #swarm_model.update_stream_x(np.concatenate((np.array(state[0:6,0],ndmin=2).transpose(),centroid, centroid_v, np.array(cmd[0:3,0],ndmin=2).transpose()),axis=0))

    # Evolve the states
    # -----------------
    state = node.evolve(Ts, state, cmd)
    #state = node.evolve_sat(Ts, state, cmd)
     
    # Store results
    # -------------
    t_all[i]                = t
    states_all[i,:,:]       = state
    cmds_all[i,:,:]         = cmd
    targets_all[i,:,:]      = targets
    obstacles_all[i,:,:]    = obstacles
    centroid_all[i,:,:]     = centroid
    f_all[i]                = f
    lemni_all[i,:]          = lemni
    metrics_order_all[i,:]  = metrics_order
    pins_all[i,:,:]         = pin_matrix  
    
    # Increment 
    # ---------
    t += Ts
    i += 1
        
    #%% Compute Trajectory
    # --------------------
     
    # EXPIRMENT # 1 - exclude random agents from the swarm every 10 seconds
    # select which agents to exclude (manually)
    # every 10 seconds
    #if t%10 < Ts:
        # randomly exclude
        #exclusion = [random.randint(0, nVeh-1)]
        #print(exclusion)
        
    # # EXPIRMENT # 2 - manually exclude agents from the swarm
    # # for simulation
    # if t < 20:
    #     exclusion = [2]
    # if t > 20 and t <= 45:
    #     exclusion = [2,7]
    # if t > 45 and t <= 65:
    #     exclusion = [1]
    # if t > 45 and t <= 75:
    #     exclusion = [1,6]
    # if t > 75 and t <= 90:
    #     exclusion = [3]
    # if t > 75 and t <= 100:
    #     exclusion = [3,7]
    # if t > 100 and t <= 115:
    #     exclusion = [9]
    # if t > 115 and t <= 120:
    #     exclusion = [5]
    # if t > 120 and t <= 150:
    #     exclusion = [4]
    # if t > 150 and t <= 185:
    #     exclusion = [4,8]
    # if t > 185 and t <= 200:
    #     exclusion = [6]
        
    # # Experiment #3 - remove 2 then remove 2
    # if t < 20:
    #     exclusion = []
    # if t > 20 and t <= 50:
    #     exclusion = [1]
    # if t > 50 and t <= 80:
    #     exclusion = [1,2]
    # if t > 80 and t <= 110:
    #     exclusion = [2]
    # if t > 110 and t <= 140:
    #     exclusion = []

           
    # create a temp exlusionary set
    state_ = np.delete(state, [exclusion], axis = 1)
    targets_ = np.delete(targets, [exclusion], axis = 1)
    lemni_all_ = np.delete(lemni_all, [exclusion], axis = 1)
        
    #if flocking
    if tactic_type == 'reynolds' or tactic_type == 'saber' or tactic_type == 'starling' or tactic_type == 'pinning':
        trajectory = targets 
    
    # if encircling
    if tactic_type == 'circle':
        #trajectory, _ = encircle_tools.encircle_target(targets, state)
        trajectory, _ = encircle_tools.encircle_target(targets_, state_)
    
    # if lemniscating
    elif tactic_type == 'lemni':
        #trajectory, lemni = lemni_tools.lemni_target(lemni_all,state,targets,i,t)
        trajectory, lemni = lemni_tools.lemni_target(lemni_all_,state_,targets_,i,t)

    # add exluded back in
    for ii in exclusion:
        trajectory = np.insert(trajectory,ii,targets[:,ii],axis = 1)
        trajectory[0:2,ii] = ii + 5 # just move away from the swarm
        trajectory[2,ii] = 15 + ii 
        lemni = np.insert(lemni,ii,lemni_all[i-1,ii],axis = 1)
        # label excluded as pins (for plotting colors only)
        pins_all[i-1,ii,ii] = 1       
            
    #%% Prep for compute commands (next step)
    # ----------------------------
    states_q = state[0:3,:]     # positions
    states_p = state[3:6,:]     # velocities 
    
    # Compute metrics
    # ---------------
    centroid                = tools.centroid(state[0:3,:].transpose())
    centroid_v              = tools.centroid(state[3:6,:].transpose())
    swarm_prox              = tools.sigma_norm(centroid.ravel()-targets[0:3,0])
    metrics_order[0,0]      = swarm_metrics.order(states_p)
    metrics_order[0,1:7]    = swarm_metrics.separation(states_q,targets[0:3,:],obstacles)
    metrics_order[0,7:9]    = swarm_metrics.energy(cmd)
    metrics_order[0,9:12]   = swarm_metrics.spacing(states_q)
        
    # load the updated centroid states (x,v)
    # ---------------------------------------
    #swarm_model.update_stream_y(np.concatenate((np.array(state[0:6,0],ndmin=2).transpose(),centroid, centroid_v),axis=0))
    #if swarm_model.count_y >= swarm_model.desired_size:
    #    swarm_model.fit()
    #    swarm_model.count_x    = -1
    #    swarm_model.count_y    = -1

    # Add other vehicles as obstacles (optional, default = 0)
    # -------------------------------------------------------
    if vehObs == 0: 
        obstacles_plus = obstacles
    elif vehObs == 1:
        states_plus = np.vstack((state[0:3,:], rVeh*np.ones((1,state.shape[1])))) 
        obstacles_plus = np.hstack((obstacles, states_plus))
            
    #%% Compute the commads (next step)
    # --------------------------------       
    cmd, params, pin_matrix = tactic.commands(states_q, states_p, obstacles_plus, walls, targets[0:3,:], targets[3:6,:], trajectory[0:3,:], trajectory[3:6,:], swarm_prox, tactic_type, centroid, params)
       
#%% Produce animation of simulation
# ---------------------------------
#print('here1')
showObs     = 0 # (0 = don't show obstacles, 1 = show obstacles, 2 = show obstacles + floors/walls)
# show_B_max  = 1 # highlight the max influencer? (0 = np, 1 = yes)
# if tactic_type == 'pinning' and show_B_max == 1:
#     # find the max influencer in the graph
#     G = graph_tools.build_graph(states_q, 5.1)
#     B = graph_tools.betweenness(G)
#     B_max = max(B, key=B.get)
#     pins_all[:,B_max,B_max] = 2
#ani = animation.animateMe(Ts, t_all, states_all, cmds_all, targets_all[:,0:3,:], obstacles_all, walls_plots, showObs, centroid_all, f_all, tactic_type, pins_all)    
ani = animation.animateMe(Ts, t_all, states_all, cmds_all, targets_all[:,0:3,:], obstacles_all, walls_plots, showObs, centroid_all, f_all, tactic_type, pins_all)    
    
#%% Produce plots
# --------------

# separtion 
fig, ax = plt.subplots()
ax.plot(t_all[4::],metrics_order_all[4::,1],'-b')
ax.plot(t_all[4::],metrics_order_all[4::,5],':b')
ax.plot(t_all[4::],metrics_order_all[4::,6],':b')
ax.fill_between(t_all[4::], metrics_order_all[4::,5], metrics_order_all[4::,6], color = 'blue', alpha = 0.1)
#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', ylabel='Mean Distance (with Min/Max Bounds) [m]',
       title='Separation between Agents')
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
ax.grid()
plt.show()

# radii from target
radii = np.zeros([states_all.shape[2],states_all.shape[0]])
for i in range(0,states_all.shape[0]):
    for j in range(0,states_all.shape[2]):
        radii[j,i] = np.linalg.norm(states_all[i,:,j] - targets_all[i,:,j])
        
fig, ax = plt.subplots()
for j in range(0,states_all.shape[2]):
    ax.plot(t_all[4::],radii[j,4::].ravel(),'-b')
ax.set(xlabel='Time [s]', ylabel='Distance from Target for Each Agent [m]',
       title='Distance from Target')
plt.axhline(y = 5, color = 'k', linestyle = '--')
plt.show()

#%% Save data
# -----------
pickle_out = open("Data/t_all.pickle","wb")
pickle.dump(t_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/cmds_all.pickle","wb")
pickle.dump(cmds_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/states_all.pickle","wb")
pickle.dump(states_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/targets_all.pickle","wb")
pickle.dump(targets_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/obstacles_all.pickle","wb")
pickle.dump(obstacles_all, pickle_out)
pickle_out.close()
pickle_out = open("Data/centroid_all.pickle","wb")
pickle.dump(centroid_all, pickle_out)
pickle_out = open("Data/lemni_all.pickle","wb")
pickle.dump(lemni_all, pickle_out)
pickle_out.close()



