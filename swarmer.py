#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 19:07:07 2023

@author: tjards
"""

# import stuff
# ------------
import numpy as np
from utils import tools # do I really need this module?


class Agents:
    
    def __init__(self,tactic_type,nVeh):
        
        # initite attributes 
        # ------------------
        self.nVeh    =   nVeh      # number of vehicles
        self.rVeh    =   0.5     # physical radius of vehicle
        self.tactic_type = tactic_type    
                        # reynolds  = Reynolds flocking + Olfati-Saber obstacle
                        # saber     = Olfati-Saber flocking
                        # starling  = swarm like starlings 
                        # circle    = encirclement
                        # lemni     = dynamic lemniscates and other closed curves
                        # pinning   = pinning control
                        # shep      = shepherding
                        
        # Vehicles states
        # ---------------
        iSpread =   20      # initial spread of vehicles
        self.state = np.zeros((6,self.nVeh))
        self.state[0,:] = iSpread*(np.random.rand(1,self.nVeh)-0.5)                   # position (x)
        self.state[1,:] = iSpread*(np.random.rand(1,self.nVeh)-0.5)                   # position (y)
        self.state[2,:] = np.maximum((iSpread*np.random.rand(1,self.nVeh)-0.5),2)+15  # position (z)
        self.state[3,:] = 0                                                      # velocity (vx)
        self.state[4,:] = 0                                                      # velocity (vy)
        self.state[5,:] = 0                                                      # velocity (vz)
        self.centroid = tools.centroid(self.state[0:3,:].transpose())
        self.centroid_v = tools.centroid(self.state[3:6,:].transpose())
        # select a pin (for pinning control)
        self.pin_matrix = np.zeros((self.nVeh,self.nVeh))
        
        if self.tactic_type == 'pinning':
            from utils import pinning_tools
            self.pin_matrix = pinning_tools.select_pins_components(self.state[0:3,:])

        # Commands
        # --------
        self.cmd = np.zeros((3,self.nVeh))
        self.cmd[0] = np.random.rand(1,self.nVeh)-0.5      # command (x)
        self.cmd[1] = np.random.rand(1,self.nVeh)-0.5      # command (y)
        self.cmd[2] = np.random.rand(1,self.nVeh)-0.5      # command (z)

        # Other Parameters
        # ----------------
        self.params = np.zeros((4,self.nVeh))  # store dynamic parameters
        self.lemni                   = np.zeros([1, self.nVeh])

        
        
    def evolve(self, Ts):
        
        # constraints
        #vmax = 1000
        #vmin = -1000

        #discretized doubple integrator 
        self.state[0:3,:] = self.state[0:3,:] + self.state[3:6,:]*Ts
        self.state[3:6,:] = self.state[3:6,:] + self.cmd[:,:]*Ts
        #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, -vmax), vmax)
        #state[3:6,:] = np.minimum(np.maximum(state[3:6,:] + cmd[:,:]*Ts, vmin), vmax)
        #state[3:6,:] = clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax)
        #state[3:6,:] = clamp_norm_min(clamp_norm(state[3:6,:] + cmd[:,:]*Ts,vmax),vmin)
        
        
        

class Targets:

    def __init__(self, tspeed, nVeh):
        
        self.tSpeed  =   tspeed       # speed of target
        
        self.targets = 4*(np.random.rand(6,nVeh)-0.5)
        self.targets[0,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[1,:] = 0 #5*(np.random.rand(1,nVeh)-0.5)
        self.targets[2,:] = 15
        self.targets[3,:] = 0
        self.targets[4,:] = 0
        self.targets[5,:] = 0
        
        self.trajectory = self.targets.copy()
        
    def evolve(self, t):
        
        self.targets[0,:] = 100*np.sin(self.tSpeed*t)                 # targets[0,:] + tSpeed*0.002
        self.targets[1,:] = 100*np.sin(self.tSpeed*t)*np.cos(self.tSpeed*t)  # targets[1,:] + tSpeed*0.005
        self.targets[2,:] = 100*np.sin(self.tSpeed*t)*np.sin(self.tSpeed*t)+15  # targets[2,:] + tSpeed*0.0005
        
class Obstacles:
    
    def __init__(self,tactic_type,nObs,targets):
        
        # note: don't let pass-in of walls yet, as it is a manual process still
        
        # initiate attributes
        # -------------------
        
        self.nObs    = nObs     # number of obstacles 
        self.vehObs  = 0     # include other vehicles as obstacles [0 = no, 1 = yes] 

        # if using reynolds, need make target an obstacle 
        if tactic_type == 'reynolds':
            self.targetObs = 1
        else:
            self.targetObs = 0   

        # there are no obstacle, but we need to make target an obstacle 
        if self.nObs == 0 and self.targetObs == 1:
            self.nObs = 1

        self.obstacles = np.zeros((4,self.nObs))
        oSpread = 10

        # manual (comment out if random)
        # obstacles[0,:] = 0    # position (x)
        # obstacles[1,:] = 0    # position (y)
        # obstacles[2,:] = 0    # position (z)
        # obstacles[3,:] = 0

        #random (comment this out if manual)
        if self.nObs != 0:
            self.obstacles[0,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[0,0]                   # position (x)
            self.obstacles[1,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[1,0]                   # position (y)
            self.obstacles[2,:] = oSpread*(np.random.rand(1,self.nObs)-0.5)+targets[2,0]                  # position (z)
            #obstacles[2,:] = np.maximum(oSpread*(np.random.rand(1,nObs)-0.5),14)     # position (z)
            self.obstacles[3,:] = np.random.rand(1,self.nObs)+1                             # radii of obstacle(s)

        # manually make the first target an obstacle
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
            self.obstacles[3,0] = 2              # radii of obstacle(s)

        # Walls/Floors 
        # - these are defined manually as planes
        # --------------------------------------   
        self.nWalls = 1                      # default 1, as the ground is an obstacle 
        self.walls = np.zeros((6,self.nWalls)) 
        self.walls_plots = np.zeros((4,self.nWalls))

        # add the ground at z = 0:
        newWall0, newWall_plots0 = tools.buildWall('horizontal', -2) 

        # load the ground into constraints   
        self.walls[:,0] = newWall0[:,0]
        self.walls_plots[:,0] = newWall_plots0[:,0]
        
        self.obstacles_plus = self.obstacles.copy()
        
        
    def evolve(self, targets):
        
        if self.targetObs == 1:
            self.obstacles[0,0] = targets[0,0]     # position (x)
            self.obstacles[1,0] = targets[1,0]     # position (y)
            self.obstacles[2,0] = targets[2,0]     # position (z)
        
class History:
    
    # note: break out the Metrics stuff int another class 
    
    def __init__(self, Agents, Targets, Obstacles, Ts, Tf, Ti, f):
        
        nSteps = int(Tf/Ts+1)
        
        # initialize a bunch of storage 
        self.t_all               = np.zeros(nSteps)
        self.states_all          = np.zeros([nSteps, len(Agents.state), Agents.nVeh])
        self.cmds_all            = np.zeros([nSteps, len(Agents.cmd), Agents.nVeh])
        self.targets_all         = np.zeros([nSteps, len(Targets.targets), Agents.nVeh])
        self.obstacles_all       = np.zeros([nSteps, len(Obstacles.obstacles), Obstacles.nObs])
        self.centroid_all        = np.zeros([nSteps, len(Agents.centroid), 1])
        self.f_all               = np.ones(nSteps)
        self.lemni_all           = np.zeros([nSteps, Agents.nVeh])
        # metrics_order_all   = np.zeros((nSteps,7))
        # metrics_order       = np.zeros((1,7))
        nMetrics            = 12 # there are 11 positions being used.    
        self.metrics_order_all   = np.zeros((nSteps,nMetrics))
        self.metrics_order       = np.zeros((1,nMetrics))
        self.pins_all            = np.zeros([nSteps, Agents.nVeh, Agents.nVeh]) 
        # note: for pinning control, pins denote pins as a 1
        # also used in lemni to denote membership in swarm as 0

        # store the initial conditions
        self.t_all[0]                = Ti
        self.states_all[0,:,:]       = Agents.state
        self.cmds_all[0,:,:]         = Agents.cmd
        self.targets_all[0,:,:]      = Targets.targets
        self.obstacles_all[0,:,:]    = Obstacles.obstacles
        self.centroid_all[0,:,:]     = Agents.centroid
        self.f_all[0]                = f
        self.metrics_order_all[0,:]  = self.metrics_order
        #self.lemni                   = np.zeros([1, Agents.nVeh])
        self.lemni_all[0,:]          = Agents.lemni
        self.pins_all[0,:,:]         = Agents.pin_matrix     
        

        