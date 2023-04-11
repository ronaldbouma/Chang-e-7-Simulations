#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:27:09 2022

@author: Ronald
"""

#script to estimate the delta v needed for orbital insertion and plane change

import numpy as np
import matplotlib.pyplot as plt

def V(a,e=0,u=4.9048695e3,escape=False):
    #delta v uses the moons gravitational paramater in km^3
    r = a*(1-e)
    if escape==False:
        return np.sqrt(u*(2/r-1/a))
    if escape==True:
        return -np.sqrt(u*(2/r-1/a))+np.sqrt(u*(2/r))
    
def di(V1,V2,di):
    return np.sqrt(V1**2+V2**2-2*V1*V2*np.cos(di))

a_space = np.linspace(3500,1700,200)

ecc = 0.04

dv = V(a_space,e=ecc,escape=True)

i_range = np.linspace(0,50,200)
radian_range = np.radians(i_range)
velocity_2338 = V(2338,e=0.04)
velocity_3476 = V(3476,e=0.04)
velocity_1937 = V(1937,e=0.04)

dvi_2338 = di(velocity_2338,velocity_2338,radian_range)
dvi_3476 = di(velocity_3476,velocity_3476,radian_range)
dvi_1937 = di(velocity_1937,velocity_1937,radian_range)


fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(a_space,dv)
plt.title('\u0394v needed for orbital insertion')
plt.gca().invert_xaxis()
plt.ylim([0,1.5])
plt.xlabel('Semi Major axis in km')
plt.ylabel('\u0394v in km/s')

plt.subplot(2,1,2)
plt.plot(i_range,dvi_2338,label='a=2338 km, e=0.04')
plt.plot(i_range,dvi_3476,label='a=3476 km, e=0.04')
plt.plot(i_range,dvi_1937,label='a=1937 km, e=0.04')
plt.ylim([0,1.5])
plt.xlabel('inclination change in degrees')
plt.ylabel('\u0394v in km/s')
plt.legend()
plt.show()