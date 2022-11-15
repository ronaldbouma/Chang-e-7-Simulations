#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:42:41 2022

@author: Ronald
"""

#cd
import os
os.chdir('desktop/thesis/scripts')

#packages
import numpy as np
import functions2 as funcs
import matplotlib.pyplot as plt
import healpy as hp
import pygdsm

##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

##sat states
sat1_state = [2338,0.04,np.radians(70),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(80),np.pi,np.pi,0.3*np.pi]
ground_state = np.radians(170),np.radians(10)
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])

#time selection
max_years = 2
baseline_amount = [5000,10000,100000]
t_select = np.sort(np.random.randint(0,max_years*31556926,baseline_amount))
t_line = np.linspace(0,2000,5000)
t_int = 0.1 #integration time per measurement

#creating baselines, orientations and positions
Ds = funcs.baselines(t_select, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien,sat1_mut=mutation1,sat2_mut=mutation2)

Baselines = Ds[:3]
Positions = Ds[3:6]
orientations = Ds[6:]

##frequency and antennas
frequency = np.array([10e6,15e6,20e6,25e6,30e6,50e6])
wavelength = c/frequency

bandwidth = 100e3

antenna_length = 5

##make model
direction = np.radians([120,120])
size_ang = np.radians(2)
intensity = 100
intensities = intensity*(frequency/frequency[0])**(-0.5)

model_I_jansky, model_x = funcs.make_model(2**7, frequency[0], object_intensity=intensity,object_size=size_ang,object_direction=direction)

#Noise, system noise canb be added in sampling function as a fraction of sky noise
Noise_rms = np.array([])
for i in range(len(wavelength)):
    Noise_rms = np.append(Noise_rms,funcs.rms_sky(wavelength[i],bandwidth=bandwidth,t=t_int,A_length=antenna_length))




