#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:31:50 2019
@author: farismismar
"""

import math
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

# An attempt to follow
# https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py

# Environment parameters
# cell radius
# UE movement speed
# BS max tx power
# BS antenna
# UE noise figure
# Center frequency
# Transmit antenna isotropic gain
# Antenna heights
# Shadow fading margin
# Number of ULA antenna elements
# Oversampling factor

class radio_environment:
    '''    
        Observation: 
            Type: Box(8)
            Num Observation        Min      Max
            0   UE BS server       0        N-1  <- not used actually.
            1   UE BS neighb       0        N-1  <- not used actually.
            2   User server X     -r        r
            3   User server Y     -r        r
            4   User server X     isd-r     isd+r
            5   User server Y     -r        r
            6   Serving BS Power   1        40W
            7   Neighbor BS power  1        40W
            
        Actions:
            Type: Discrete(8)
            Num	Action
            0	Power up by 1 dB
            1	Power up by 3 dB
            2   Power down by 3 dB
            3   Power down by 1 dB
            4   Power neighbor down by 1 dB
            5   Power neighbor down by 3 dB
            6   Power neighbor up by 1 dB
            7   Power neighbor up by 3 dB
            
    '''     
    def __init__(self, seed):
        self.cell_radius = 350 # in meters.
        self.inter_site_distance = 3 * self.cell_radius / 2.
        self.num_users = 30 # number of users.
        self.sinr_target = 2 # in dB
        self.min_sinr = -4 # in dB
        self.max_tx_power = 40 # in Watts
        self.max_tx_interference = 40 # in Watts
        self.f_c = 2100e6 # Hz
        self.prob_LOS = 0.3 # Probability of LOS transmission
        
        self.num_actions = 8
        
        # for Beamforming
        self.M_ULA = 64
        self.k_oversampling = 2
        
        self.reward_min = -2
        self.reward_max = 100
        
        bounds_lower = np.array([
            -self.cell_radius,
            -self.cell_radius,
            self.inter_site_distance-self.cell_radius,
            -self.cell_radius,
            1,
            1])

        bounds_upper = np.array([
            self.cell_radius,
            self.cell_radius,
            self.inter_site_distance+self.cell_radius,
            self.cell_radius,
            self.max_tx_power,
            self.max_tx_interference])

        self.action_space = spaces.Discrete(self.num_actions) # action size is here
        self.observation_space = spaces.Box(bounds_lower, bounds_upper, dtype=np.float32) # spaces.Discrete(2) # state size is here 
        
        self.seed(seed=seed)
        
        self.state = None
#        self.steps_beyond_done = None
        self.received_sinr_dB = None
        self.serving_transmit_power_dB = None
        self.interfering_transmit_power_dB = None
      
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def reset(self):
        self.state = [self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=self.inter_site_distance-self.cell_radius, high=self.inter_site_distance+self.cell_radius),
                      self.np_random.uniform(low=-self.cell_radius, high=self.cell_radius),
                      self.np_random.uniform(low=1, high=self.max_tx_power),
                      self.np_random.uniform(low=1, high=self.max_tx_interference)
                      ]
        
#        self.steps_beyond_done = None
        return np.array(self.state)
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        reward = 0
        x, y, x_int, y_int, pt_serving, pt_interferer = state

        # based on the action make your call
        if (action == 0):
            pt_serving *= 10**(1/10.)
            reward += 1
        elif (action == 1):
            pt_serving *= 10**(3/10.)
            reward += 1
        elif (action == 2):
            pt_serving *= 10**(-3/10.)
            reward += 1
        elif (action == 3):
            pt_serving *= 10**(-1/10.)
            reward += 1
        elif (action == 4):
            pt_interferer *= 10**(-1/10.)
            reward += 2
        elif (action == 5):
            pt_interferer *= 10**(-3/10.)
            reward += 2
        elif (action == 6):
            pt_interferer *= 10**(1/10.)
            reward += 2
        elif (action == 7):
            pt_interferer *= 10**(3/10.)
            reward += 2
         
        if (action > self.num_actions - 1):
            print('WARNING: Invalid action played!')
            reward = 0
            return [], 0, False, True
        
        # move the UEs at a speed of v, in a random direction
        v = 5 # km/h.

        v *= 5./18 # in m/sec
        theta_1, theta_2 = self.np_random.uniform(low=-math.pi, high=math.pi, size=2)
        
        dx_1 = v * math.cos(theta_1)
        dy_1 = v * math.sin(theta_1)

        dx_2 = v * math.cos(theta_2)
        dy_2 = v * math.sin(theta_2)
        
        x += dx_1
        y += dy_1
        
        x_int += dx_2
        y_int += dy_2
        
        received_power, interference_power, received_sinr = self._compute_rf(x, y, pt_serving, pt_interferer)
        received_power_ue2, interference_power_ue2, received_ue2_sinr = self._compute_rf_user_2(x_int, y_int, pt_interferer, pt_serving) # the serving cell became interf. so did the PL
        
        # keep track of quantities...
        self.received_sinr_dB = received_sinr 
        self.received_ue2_sinr_dB = received_ue2_sinr
        self.serving_transmit_power_dBm = 10*np.log10(pt_serving*1e3)
        self.interfering_transmit_power_dBm = 10*np.log10(pt_interferer*1e3)

        # Did we find a FEASIBLE NON-DEGENERATE solution?
        done = (received_sinr >= self.sinr_target) and (pt_serving <= self.max_tx_power) and (pt_serving >= 0) and \
                (pt_interferer <= self.max_tx_interference) and (pt_interferer >= 0) and (received_ue2_sinr >= self.sinr_target)
                
        abort = (pt_serving > self.max_tx_power) or (pt_interferer > self.max_tx_interference) or \
                        (received_sinr <= self.min_sinr) or (received_ue2_sinr <= self.min_sinr) or (received_sinr > 30) or (received_ue2_sinr > 30) # SINR > 30 dB is unrealistic.
                
        print('{:.2f} dB | {:.2f} dB | {:.2f} W | {:.2f} W '.format(received_sinr, received_ue2_sinr, pt_serving, pt_interferer), end='')
#        print('Done: {}'.format(done))
#        print('UE moved to ({0:0.3f},{1:0.3f}) and their received SINR became {2:0.3f} dB.'.format(x,y,received_sinr))
        
        # Update the state.
        self.state = (x,y, x_int, y_int, pt_serving, pt_interferer)
     
        if done:
            reward += self.reward_max
        if abort:
            reward = self.reward_min
            
        return np.array(self.state), reward, done, abort

    def _compute_bf_vector(self, theta_n):
        c = 3e8 # speed of light
        wavelength = c / self.f_c
        
        d = wavelength / 2. # antenna spacing 
        k = 2. * math.pi / wavelength
        
        exponent = 1j * k * d * math.cos(theta_n)
        
        f = 1. / math.sqrt(self.M_ULA) * np.exp(exponent * np.arange(self.M_ULA, dtype=complex))
        
        return f

    # TODO: build the same function for the second user
    def _compute_channel(self, Np, theta, x_ue, y_ue, x_bs, y_bs):
        # Np is the number of paths p
        # theta is the steering angle

        path_loss = self._path_loss(x_ue, y_ue, x_bs, y_bs)

        h = 0 + 0j
        alpha = np.ones(Np) * self.prob_LOS * 1 + (1 - self.prob_LOS) * np.random.normal()
        a_theta = self._compute_bf_vector(theta)
        for p in np.arange(Np):
            h += alpha[p] * a_theta.T
        
        h *= math.sqrt(self.M_ULA) / path_loss
        
        return h
        
    def _compute_rf(self, x, y, pt_serving, pt_interferer):
        # Without loss of generality, the base station is at the origin
        # The interfering base station is x = cell_radius, y = 0
        x_bs_int = self.inter_site_distance
        y_bs_int = 0
        
        ue_noise_figure = 7 # dB
        T = 290 # Kelvins
        B = 15000 # Hz
        k = 1.38e-23
        g_ant = 17 # dBi
        NRB = 25 # 5 MHz channel

        # Find the received powers
        path_loss = self._path_loss(x,y)
        int_path_loss = self._path_loss(x,y,x_bs_int,y_bs_int)
        
        noise_power = k*T*B*10**(ue_noise_figure/10.) # Watts
        interference_power = pt_interferer / NRB * 10**(-int_path_loss/10.) * 10**(g_ant/10.) # Watts
        
        received_power = 10*np.log10(pt_serving*1e3) -10*np.log(NRB) + g_ant - path_loss
        interference_plus_noise_power = 10*np.log10((noise_power + interference_power)*1e3)

#        print(received_power, interference_plus_noise_power) # before fading, ok
        
        # TODO: Find the Rician coefficient formula and add it for the LOS component.
        
        # Add Rayleigh fading and log-normal fading here.
        h = np.random.normal(size=2) / math.sqrt(2)
        rayleigh_coeff = (1 - self.prob_LOS) * 10*np.log10(np.linalg.norm(h) ** 2) + self.prob_LOS * 0.
        sf_margin = 2 # dB
        log_normal_fading = np.random.normal(0, sf_margin)
        received_power -= rayleigh_coeff + log_normal_fading
        
        h = np.random.normal(size=2) / math.sqrt(2)
        rayleigh_coeff = (1 - self.prob_LOS) * 10*np.log10(np.linalg.norm(h) ** 2) + self.prob_LOS * 0.
        sf_margin = 2 # dB
        log_normal_fading = np.random.normal(0, sf_margin)
        interference_plus_noise_power -= rayleigh_coeff + log_normal_fading
        
        received_sinr = received_power - interference_plus_noise_power
        
#        print(pt_serving)

        
#        print(pt_serving, received_sinr) # after fading, ?
                
       # print(10*np.log10(1e3*pt_serving), path_loss, received_power)  # this all ok.


        return [received_power, interference_power, received_sinr]
    
    def _compute_rf_user_2(self, x_int, y_int, pt_serving, pt_interferer):
        # Without loss of generality, the base station is at the origin
        # The interfering base station is x = cell_radius, y = 0
        x_bs_int = self.inter_site_distance
        y_bs_int = 0
        
        ue_noise_figure = 7 # dB
        T = 290 # Kelvins
        B = 15000 # Hz
        k = 1.38e-23
        g_ant = 17 # dBi
        NRB = 25 # 5 MHz channel
        
         # Find the received powers
        path_loss = self._path_loss(x_int, y_int, x_bs_int, y_bs_int) # this is the pathloss from bs_int to ue_2
        int_path_loss = self._path_loss(x_int, y_int) # pathloss from bs to ue_2
        
        noise_power = k*T*B*10**(ue_noise_figure/10.) # Watts
        interference_power = pt_interferer / NRB * 10**(-int_path_loss/10.) * 10**(g_ant/10.) # Watts
        
        received_power = 10*np.log10(pt_serving*1e3) -10*np.log(NRB) + g_ant - path_loss
        interference_plus_noise_power = 10*np.log10((noise_power + interference_power)*1e3)

        # Add Rayleigh fading and log-normal fading here.
        h = np.random.normal(size=2) / math.sqrt(2)
        rayleigh_coeff = 10*np.log10(np.linalg.norm(h)) ** 2
        sf_margin = 2 # dB
        log_normal_fading = np.random.normal(0, sf_margin)        
        received_power -= rayleigh_coeff + log_normal_fading
        
        h = np.random.normal(size=2) / math.sqrt(2)
        rayleigh_coeff = 10*np.log10(np.linalg.norm(h)) ** 2
        sf_margin = 2 # dB
        log_normal_fading = np.random.normal(0, sf_margin)
        interference_plus_noise_power -= rayleigh_coeff + log_normal_fading
        
        received_sinr = received_power - interference_plus_noise_power

        return [received_power, interference_power, received_sinr]
    
    def _path_loss(self, x, y, x_bs=0, y_bs=0):
        f_c = self.f_c
        c = 3e8 # speed of light
        d = math.sqrt((x - x_bs)**2 + (y - y_bs)**2)
        h_B = 20
        h_R = 1.5

#        print('Distance from cell site is: {} km'.format(d/1000.))
        # FSPL
        L_fspl = -10*np.log10((4.*math.pi*c/f_c / d) ** 2)
        
        # COST231        
        C = 3
        a = (1.1 * np.log10(f_c/1e6) - 0.7)*h_R - (1.56*np.log10(f_c/1e6) - 0.8)
        L_cost231  = 46.3 + 33.9 * np.log10(f_c/1e6) + 13.82 * np.log10(h_B) - a + (44.9 - 6.55 * np.log10(h_B)) * np.log10(d/1000.) + C
    
        L = L_cost231
        
        return L # in dB