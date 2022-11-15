#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:35:03 2022

@author: Ronald
"""

#cd
import os
import sys


#packages
import numpy as np
import functions2 as funcs
import healpy as hp
import time

##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

#select frequency list
arguments = len(sys.argv)
if arguments == 2:
    select = int(sys.argv[1])
    if select==1:
        frequency_list = 'low'
    elif select==2:
        frequency_list = 'med'
    elif select == 3:
    	frequency_list = 'high'
    elif select in range(10,50,1):
    	frequency_list = None
    	frequency_select = select
    else:
        print('improper argument for frequency selection, 1 for low, 2 for high, or input between 10 and 50 for that frequency in Mhz, leave empty for full list')
else:
    frequency_list = 'full'

##sat states
sat1_state = [2338,0.04,np.radians(70),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(80),np.pi,np.pi,0.3*np.pi]
ground_state = np.radians(170),np.radians(10)
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])


##frequency and antennas
if frequency_list == 'full':
    frequency = np.array([10e6,15e6,20e6,25e6,30e6,50e6])
if frequency_list == 'low':
    frequency = np.array([10e6,15e6])
if frequency_list == 'med':
	frequency = np.array([20e6,25e6])
if frequency_list == 'high':
    frequency = np.array([30e6,50e6])
if frequency_list == None:
	frequency = frequency_select * 1e6

print('frequencies are: {} MHz'.format(frequency))

wavelength = c/frequency

bandwidth = 100e3

antenna_length = 5

#model parameters
NSIDE = 2**7
NPIX = hp.nside2npix(NSIDE)
##location of fossil
direction = np.radians([[110,70],[130,60],[110,40]])
size_ang = np.radians([2,2,2])
intensity_base = np.array([250,500,1000])
intensities_multiplier = (frequency/10e6)**(-0.5)


#time selection
max_years = 2
t_int = 0.1
time_span = [0,max_years*31556926]
baseline_amount = [500000,1000000,2000000]#[100000,500000,1000000,2000000] #must be integers

t_points = funcs.select_visible(time_span, direction[0], baseline_amount[-1], sat1_state, sat2_state, ground_state)


##running the simulations
print('starting simulations')
print('')
t_start = time.time()
for k in range(len(baseline_amount)):
    t_select = t_points[:baseline_amount[k]]
    Ds = funcs.baselines(t_select, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
    Positions = Ds[3:6]
    orientations = Ds[6:]
    del Ds    
    del t_select
    
    for i in range(len(frequency)):
        print('')
        print('starting simulation for {} Baselines at frequency {} MHz'.format(baseline_amount[k],int(frequency[i]/1e6)))
        print('')
        intensities_fossil = intensity_base *intensities_multiplier[i]
        model_I, model_x = funcs.make_model(NSIDE, frequency[i], object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)
        noise_sky = funcs.rms_sky(wavelength[i],bandwidth=bandwidth, t=t_int, A_length=antenna_length)

        if i ==0:
            Vs, Baselines,measure_map = funcs.Visibility_sampling_weclipse(Positions, model_x, model_I, wavelength[i], \
                                                  orientations=orientations,Noise=[0,noise_sky], system_noise_factor=0.1, bandwidth=bandwidth, antenna_length=5, measure_map_option=True)[:3]
            filename_map = os.path.join("output/measurement_map_B{}.npz".format(len(Baselines[0])))
            np.savez(filename_map, measure_map = measure_map)
        else:
            Vs, Baselines = funcs.Visibility_sampling_weclipse(Positions, model_x, model_I, wavelength[i], \
                                                      orientations=orientations,Noise=[0,noise_sky], system_noise_factor=0.1, bandwidth=bandwidth, antenna_length=5, measure_map_option=False)[:2]
            #cleaned = funcs.Cleaning_2(Vs, eclipse, Positions, model_x, wavelength[i], floor=1.5*intensities_fossil[i],cycles=50,points_per_cycle=int(NSIDE/4))
        reconstruction = funcs.reconstruct(Positions, Vs, NPIX, model_x, wavelength[i],divide_by_baselines=True)
        filename = os.path.join("output/F{}MHz_B{}_notcleaned.npz".format(int(frequency[i]/1e6),len(Baselines[0])))
        np.savez(filename, reconstruction = reconstruction, Vs = Vs, Positions= Positions)
        
t_end = time.time()
print('simulations completed, run time was {} seconds'.format(round(t_end-t_start)))
print('')
