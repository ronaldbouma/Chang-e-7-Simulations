#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:41:55 2022

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
from tqdm import tqdm



##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

##sat states
sat1_state = [2338,0.04,np.radians(76),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(86),np.pi,np.pi,0.3*np.pi]
sat1_state2 = [2338,0.1,np.radians(60),np.pi,np.pi,0]
sat2_state2 = [2200,0.1,np.radians(80),np.pi,np.pi,0.3*np.pi]
ground_state = np.radians(170),np.radians(10)
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])

antenna_length = 5


frequency = 30e6
wavelength = c/frequency

#time
u=4.9048695e3
period = 2*np.pi*np.sqrt(sat1_state[0]**3/u)
resolution = 0.0001
depth = 0.05
time_span = np.linspace(3,period,20)
t_points_amount = int(depth/resolution)



#
model_x = funcs.make_model(2**2, frequency)[1]

decoherence = np.zeros((len(time_span),int(t_points_amount/2),len(model_x)))


#baselines 

Baselines = np.zeros((len(time_span),t_points_amount,3))
sat1_positions = np.zeros((len(time_span),t_points_amount,3))
sat2_positions = np.zeros((len(time_span),t_points_amount,3))

for i in tqdm(range(len(time_span))):
    t_points = np.linspace(time_span[i]-0.5*depth,time_span[i]+0.5*depth,int(depth/resolution))
    Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
    sat1_positions[i] = Ds[4]
    sat2_positions[i] = Ds[5]
    Baselines[i] = Ds[2] 
    
    
phases = np.zeros((len(time_span),len(model_x),t_points_amount))
Vs = np.zeros((len(time_span),len(model_x),t_points_amount),dtype=np.complex128)

for j in tqdm(range(len(time_span))):
    D = Baselines[j]/wavelength
    D_dot_sig = np.matmul(model_x,np.transpose(D))  
    Vs[j] = np.cos(2*np.pi*D_dot_sig) + 1j*np.sin(2*np.pi*D_dot_sig)
    phases[j] = np.angle(Vs[j])




for k in range(len(time_span)):
    midpoint = int(len(Vs[0][0])/2)
    real_I = abs(Vs[k,:,int(len(Vs[0][0])/2)])
    for g in range(int(t_points_amount/2)):
        points = np.arange(midpoint-g,midpoint+g+1)
        summed_V = np.sum(Vs[k,:,points],axis=0)
        summed_abs_V = abs(summed_V)/len(points)
        decoherence[k,g] = summed_abs_V
        
t_int = np.linspace(resolution, depth,len(decoherence[0]))

time_select = (0,4,10,15,19)

select = np.array((1,int(len(model_x)/4),int(len(model_x)/3),int(len(model_x)/2),int(len(model_x)/1.5),int(len(model_x)-1)))
lonlat = np.round(funcs.cart_to_sph(model_x[select],lat=True))




for h in range(len(time_select)):
    plt.plot(t_int,(decoherence[time_select[h],:,select[0]]),label='lat={}, lon={}'.format(lonlat[0][0],lonlat[0][1]))
    plt.plot(t_int,(decoherence[time_select[h],:,select[1]]),label='lat={}, lon={}'.format(lonlat[1][0],lonlat[1][1]))
    plt.plot(t_int,(decoherence[time_select[h],:,select[2]]),label='lat={}, lon={}'.format(lonlat[2][0],lonlat[2][1]))
    plt.plot(t_int,(decoherence[time_select[h],:,select[3]]),label='lat={}, lon={}'.format(lonlat[3][0],lonlat[3][1]))
    plt.plot(t_int,(decoherence[time_select[h],:,select[4]]),label='lat={}, lon={}'.format(lonlat[4][0],lonlat[4][1]))
    plt.plot(t_int,(decoherence[time_select[h],:,select[5]]),label='lat={}, lon={}'.format(lonlat[5][0],lonlat[5][1]))
    plt.xlabel('time in seconds')
    plt.ylabel('decoherence')
    plt.title('Decoherence at {} Mhz'.format(frequency/1e6))
    plt.legend()
    plt.show()




hp.mollview(decoherence[2,-1,:],title='Decoherence after {} seconds ({} Mhz)'.format(depth,frequency/1e6))
