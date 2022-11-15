#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:08:56 2022

@author: Ronald
"""

#cd
import os
import sys
os.chdir(os.path.expanduser('~/desktop/thesis/scripts'))

#packages
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import functions2 as funcs
from scipy.signal import peak_widths
from scipy.signal import find_peaks
from tqdm import tqdm

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

bandwidth = 100e3
antenna_length = 5



#frequency ranges
frequency = (10e6,15e6,20e6,25e6,30e6,50e6)
bin_size = 1.
bins = 600



#time selection
max_years = 2
t_int = 0.1
time_span = [0,max_years*31556926]
baseline_amount = 5000
target = np.vstack(((110,40),np.degrees(np.random.rand(10,2) * (np.pi,2*np.pi))))#(110,40)

t_points = np.sort(np.random.randint(0,time_span[1],baseline_amount))
Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
Positions = Ds[3:6]
orientations = Ds[6:]
Baselines = np.array([Positions[0] - Positions[1],Positions[0] - Positions[2],Positions[1] - Positions[2]])
Baselines = np.reshape(Baselines,(len(Baselines)*len(Baselines[0]),3))

target_xyz = hp.pixelfunc.ang2vec(np.radians(target[:,0]),np.radians(target[:,1]))
stored_min = np.zeros((len(frequency),bins+1))
stored_min_data = np.zeros((len(frequency),6))


for g in tqdm(range(len(frequency))):
    frequencies = np.linspace(frequency[g],frequency[g]+bin_size*(bins+1),bins+1,endpoint=False)
    wavelength = c/frequencies
    widths = np.zeros((len(target),len(Baselines)))
    phases = np.zeros((len(target),len(Baselines),len(frequencies)))

    for j in range(len(target)):
        for i in range(len(frequencies)):
            Ds = Baselines/wavelength[i]
            D_dot_sig = np.matmul(Ds,target_xyz[j])  
            Vs = np.cos(2*np.pi*D_dot_sig) + 1j*np.sin(2*np.pi*D_dot_sig)
            phase = np.angle(Vs)
            phases[j,:,i] = phase


    for k in range(len(target)):
        for l in range(len(phases[0])):
            peaks = find_peaks(phases[k][l])[0]
            phase_rel = phases[k][l]
            if len(peaks) == 0 or len(peaks) == 1:
                widths[k][l] = 1e6
            else:
                width = np.max(peak_widths(phase_rel,peaks)[0])
                widths[k][l] = width



    minimum_fwhm = np.min(widths,axis=1)*bin_size
    minimum_set = np.argmin(widths,axis=1)
    abs_min = np.min(minimum_fwhm)
    abs_armin = np.argmin(minimum_fwhm)
    #print(np.min(minimum_fwhm))

    minimum_phase = phases[abs_armin,minimum_set[abs_armin]]
    
    stored_min[g] = minimum_phase
    stored_min_data[g] = np.append(abs_min,np.append(Baselines[minimum_set[abs_armin]],target[abs_armin]))

print(stored_min_data[:,0])

plt.plot(frequencies-frequency[-1],stored_min[0])
plt.show()

bin_res = np.floor(np.min(stored_min/2)/10) *10
fft_size = 2**10
f_sampling = bin_res * fft_size
time_int = fft_size / f_sampling
fft_bitsize = sys.getsizeof(np.fft.fft(np.random.rand(1024)))

print('sampling needs to be at {} KHz at a fft size of {}'.format(f_sampling/1e3,fft_size))
print('')
print('bit size single fft is {} KB, per second data rate is {} MB/s'.format(fft_bitsize/(8*1024),(fft_bitsize/(8*1024))*f_sampling / 1024))


