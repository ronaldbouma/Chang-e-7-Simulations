#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 13:24:41 2021

@author: Ronald
"""
#cd
import os
os.chdir('desktop/thesis/scripts')


import numpy as np
import matplotlib.pyplot as plt
import functions as funcs


orien=[np.radians(180),np.radians(0)]
target=[np.radians(90),np.radians(0)]
target = [1.8086191330958863,3.6175915404973376]

orientation1 = [np.radians(0),np.radians(0)]
orientation2 = [np.radians(90),np.radians(180)]

theta,phi = np.linspace(0.00001,0.9999*np.pi,100),np.linspace(0,2*np.pi,100)
R = np.array([])
check1 = np.array([])
check2 =np.array([])

for i in range(len(phi)):
    for j in range(len(theta)):
        first = funcs.rot_transform(orientation1,[theta[j],phi[i]])
        second = funcs.rot_transform(orientation2,[theta[j],phi[i]])
        R = np.append(R,funcs.dipole_inter(first[0],second[0]))
        check1=np.append(check1,first)
        check2=np.append(check2,second)
    
R = np.reshape(R,(len(phi),len(theta)))


THETA, PHI = np.meshgrid(theta, phi)
Rlog = np.log10(R/np.max(R))
X, Y, Z= np.cos(PHI)*np.sin(THETA), np.sin(THETA)*np.sin(PHI), np.cos(THETA)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
colors=plt.cm.Spectral(R)
sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral)
sm.set_array(R)
fig.colorbar(sm, shrink=0.5, aspect=5)
plot = ax.plot_surface(
    X, Y, Z, rstride=1, cstride=1, 
    facecolors=colors,linewidth=0, antialiased=False, alpha=0.5)

plt.show()
