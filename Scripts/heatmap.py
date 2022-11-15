#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 11:44:52 2022

@author: Ronald
"""

#cd
import os
import sys


#packages
import numpy as np
import functions2 as funcs
import healpy as hp
from tqdm import tqdm 
import matplotlib.pyplot as plt

##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)


##sat states
sat1_state = [2338,0.04,np.radians(70),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(80),np.pi,np.pi,0.3*np.pi]
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])

ground_states = np.radians(np.array([[170,10],[140,10],[110,10],[90,0],[50,10]]))

frequency = 10e6


wavelength = c/frequency

bandwidth = 100e3

antenna_length = 5

#model parameters
NSIDE = 2**6
NPIX = hp.nside2npix(NSIDE)



#time selection
max_years = 2
t_int = 0.1
time_span = max_years*31556926
baseline_amount = 3000

time_points = np.sort(np.random.choice(time_span,baseline_amount))
measurement_map = np.zeros([len(ground_states),2,NPIX])

model_x = funcs.make_model(NSIDE, frequency)[1]

latitude = 90-np.degrees(ground_states[:,0])

for k in range(len(ground_states)):
    
    Ds = funcs.baselines(time_points, ground_states[k], sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
    Positions = Ds[3:6]
    del Ds    
    Positions=np.asarray(Positions)

    
    for i in tqdm(range(len(time_points))):
        eclipse = funcs.eclipse_checker_single(Positions[:,i,:], model_x)
        measurement_map[k][0] += np.sum(eclipse,axis=0)
        measurement_map[k][1] += np.prod(eclipse,axis=0)

    measurement_map[k][0] = measurement_map[k][0] * (100/(3*baseline_amount))
    measurement_map[k][1] = measurement_map[k][1] * (100/baseline_amount)

#for j in range(len(measurement_map)):
#    plt.title('Visibility of directions, ground station at {} latitude'.format(latitude[j]),y=1.18)
#    plt.subplot(1,2,1)
#    hp.mollview(measurement_map[j][0],hold=True,min=0,max=100, title='Single baselines')
#    #hp.graticule(dpar=60)
#    plt.subplot(1,2,2)
#    hp.mollview(measurement_map[j][1],hold=True,min=0,max=100, title='All baselines')
#    #hp.graticule(dpar=60)
#    plt.show()
    
#im = plt.imshow(measurement_map[0][0])

#plt.subplot(2,2,1)
#hp.mollview(measurement_map[0][0],hold=True,min=0,max=100,cbar=False, title='Single baselines, l=-80')
#hp.graticule(dpar=60)
#plt.subplot(2,2,2)
#hp.mollview(measurement_map[0][1],hold=True,min=0,max=100,cbar=False, title='All baselines, l=-80')
#hp.graticule(dpar=60)
#plt.subplot(2,2,3)
#hp.mollview(measurement_map[3][0],hold=True,min=0,max=100,cbar=False, title='Single baselines, l=0')
#plt.subplot(2,2,4)
#hp.mollview(measurement_map[3][1],hold=True,min=0,max=100,cbar=False, title='All baselines, l=0')

#colors = np.linspace(0,100,100)
#mappable = plt.scatter(np.zeros(100), colors, s=30, c=colors, cmap='viridis')
#plt.colorbar(mappable,orientation="vertical",fraction=0.07,anchor=(1.0,0.0))

#plt.show()

colors = np.linspace(0,100,100)
mappable = plt.scatter(np.zeros(100), colors, s=30, c=colors, cmap='viridis')

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2,nrows=2)
plt.axes(ax1)
hp.mollview(measurement_map[0][0],hold=True,min=0,max=100,cbar=False, title='Single baselines, l=-80')

plt.axes(ax2)
hp.mollview(measurement_map[0][1],hold=True,min=0,max=100,cbar=False, title='All baselines, l=-80')

plt.axes(ax3)
hp.mollview(measurement_map[3][0],hold=True,min=0,max=100,cbar=False, title='Single baselines, l=0')

plt.axes(ax4)
hp.mollview(measurement_map[3][1],hold=True,min=0,max=100,cbar=False, title='All baselines, l=0')


cbax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
cb = fig.colorbar(mappable, ax=(ax3, ax4),cax=cbax,orientation='horizontal')


plt.show()



#filename = os.path.join("output/heatmap_full.npz")
#np.savez(filename, measurement_map=measurement_map)






