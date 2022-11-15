#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 12:21:53 2022

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


#selection of paramaters
arguments = len(sys.argv)
if arguments == 5:
    frequency_arg = sys.argv[1]
    baseline_amount_arg = sys.argv[2]
    if sys.argv[3] == 'nc':
        cleaning_option = False
    elif sys.argv[3] == 'cl':
        cleaning_option = True
    else:
        print('third argument invalid, input cl for cleaning and nc for no cleaning')
    if sys.argv[4] == 'lr':
        resolution_option = 'low'
    elif sys.argv[4] == 'hr':
        resolution_option = 'high'
else:
    print('improper arguments, inputs are: frequency in Mhz, baselines in thousands, cl and nc for clean and no clean, lr and hr for low and high res')


frequency = float(frequency_arg) * 1e6
baseline_amount = float(baseline_amount_arg) * 1000


##sat states
sat1_state = [2338,0.04,np.radians(70),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(80),np.pi,np.pi,0.3*np.pi]
ground_state = np.radians(170),np.radians(10)
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])


wavelength = c/frequency

bandwidth = 100e3

antenna_length = 5

#model parameters
if resolution_option == 'low':
    NSIDE = 2**7
elif resolution_option == 'high':
    NSIDE = 2**8

NPIX = hp.nside2npix(NSIDE)
##location of fossil
direction = np.radians([[110,70],[130,60],[110,40]])
size_ang = np.radians([2,2,2])
intensity_base = np.array([75,150,300])
intensities_multiplier = (frequency/10e6)**(-0.5)


#time selection
max_years = 2
t_int = 0.1
time_span = [0,max_years*31556926]

t_points = funcs.select_visible(time_span, direction[0], int(baseline_amount), sat1_state, sat2_state, ground_state)

print('Starting simulation for {} Mhz, with {} baselines at {} resolution, cleaning is {}'.format(int(frequency/1e6),int(baseline_amount),resolution_option,cleaning_option))
print('')
t_start = time.time()

intensities_fossil = intensity_base * intensities_multiplier
model_I, model_x = funcs.make_model(NSIDE, frequency, object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)
noise_sky = funcs.rms_sky(wavelength,bandwidth=bandwidth, t=t_int, A_length=antenna_length)
Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
Positions = Ds[3:6]
orientations = Ds[6:]
del Ds
del t_points

Vs, Baselines = funcs.Visibility_sampling_weclipse(Positions, model_x, model_I, wavelength, \
                                          orientations=orientations,Noise=[0,noise_sky], system_noise_factor=0.1, bandwidth=bandwidth, antenna_length=5, bars=True)[:2]

if cleaning_option == False:
    reconstruction = funcs.reconstruct(Positions, Vs, NPIX, model_x, wavelength,divide_by_baselines=True,bars=True)
    filename = os.path.join("output/F{}MHz_B{}_notcleaned.npz".format(int(frequency/1e6),int(baseline_amount)))
    if resolution_option == 'high':
        filename = filename = os.path.join("output/hr/F{}MHz_B{}_notcleaned.npz".format(int(frequency/1e6),int(baseline_amount)))
    np.savez(filename, reconstruction = reconstruction, Vs = Vs, Positions= Positions)
    
elif cleaning_option == True:
    full_sky, Vs_clean_tot,Vs_cleaning, points_removed, pixels = funcs.Cleaning_2(Vs, Positions, model_x, wavelength, floor=1.5*intensities_fossil[0],cycles=50,points_per_cycle=int(NSIDE/4))
    
    filename = os.path.join("output/cleaned/F{}MHz_B{}.npz".format(int(frequency/1e6),int(baseline_amount)))
    if resolution_option == 'high':
        filename = os.path.join("output/cleaned/F{}MHz_B{}_hr.npz".format(int(frequency/1e6),int(baseline_amount)))
    np.savez(filename, full_sky = full_sky, Vs_clean_tot = Vs_clean_tot, Vs_cleaning=Vs_cleaning,\
             points_removed=points_removed, pixels=pixels ,Positions= Positions)
    
     
t_end = time.time()
print('simulations completed, run time was {} seconds'.format(round(t_end-t_start)))
print('')
    
    

