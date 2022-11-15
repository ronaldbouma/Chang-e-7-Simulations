#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 14:14:08 2022

@author: Ronald
"""

#cd
import os
direct = os.getcwd()
if direct != '/Users/Ronald/Desktop/Thesis/Scripts':
    os.chdir('desktop/thesis/scripts')

#packages
import numpy as np
import functions2 as funcs
import healpy as hp
import matplotlib.pyplot as plt


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

frequency = 50e6

wavelength = c/frequency

bandwidth = 100e3

antenna_length = 5


NSIDE = 2**5
NPIX = hp.nside2npix(NSIDE)

model_I, model_x = funcs.make_model(NSIDE, frequency)
model_I = np.zeros(len(model_I))

np.random.seed(1234)
sources = np.random.choice(len(model_I),8)
model_I[sources] = 1

t_points = np.random.choice(10000,2500)

Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
Positions = np.array(Ds[3:6]) / 5e3
orientations = Ds[6:]
eclipse = np.ones([3,len(t_points),NPIX],dtype=bool)
Vs, Baselines = funcs.Visibility_sampling(Positions, eclipse, model_x, model_I, wavelength, bandwidth=bandwidth, antenna_length=5, bars=True)[:2]
Vs_select = Vs[0]

Baselines_select = Baselines[0]


ring_struc = np.zeros(NPIX)


value = abs(Vs_select)
angle = np.angle(Vs_select)

figures = np.zeros([3,len(model_x)])
number = np.zeros(3)
for i in range(len(Vs_select)):
    phase_ring = funcs.beachball(Baselines_select[i], model_x, wavelength)
    ring = abs(Vs_select[i]) * np.cos(np.angle(Vs_select[i])-phase_ring)
    ring_struc += ring
    
    if i == 0:
        figures[0] = ring_struc/(i+1)
        number[0] = i+1
    elif i == 199:
        figures[1] = ring_struc/(i+1)
        number[1] = i+1
    elif i == 2499:
        figures[2] = ring_struc/(i+1)
        number[2] = i+1




fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,ncols=2)

plt.axes(ax1)
hp.mollview(model_I,cbar=False,title='input model', hold=True)

plt.axes(ax2)
hp.mollview(figures[0],cbar=False,title='{} baselines'.format(int(number[0])), hold=True)

plt.axes(ax3)
hp.mollview(figures[1],cbar=False,title='{} baselines'.format(int(number[1])), hold=True)

plt.axes(ax4)
hp.mollview(figures[2],cbar=False,title='{} baselines'.format(int(number[2])), hold=True)

plt.tight_layout()
plt.show()



