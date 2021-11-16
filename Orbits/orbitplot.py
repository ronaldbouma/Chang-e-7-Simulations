#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:03:41 2020

@author: Ronald
"""
#cd Desktop/Thesis/orbit/

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.constants
import pandas as pd

data = np.genfromtxt('unperturbedtest/unperturbedtest.dat',
                     delimiter=',')

def kepler(x,u=398600.441): #input state vector and return keplerian elements
    import numpy as np
    mu = u #gravitational parameter in km^3 units as standard unless otherwise specified
    r,v = np.array([x[0],x[1],x[2]]),np.array([x[3],x[4],x[5]]) #extract distances and velocities
    R = np.linalg.norm(r) #total distance
    V = np.linalg.norm(v) #total velocity
    rhat = r/R #hats are normal(directional) vectors
    h = np.cross(r,v)
    H = np.linalg.norm(h)
    N = np.cross([0,0,1],h)
    Nhat = N/np.linalg.norm(N)
    a = 1/(2/R - V**2/mu)
    evec = np.cross(v,h)/mu - r/R #e vector
    e = np.linalg.norm(evec) #magnitude (norm) of the e vector
    ehat = evec/e #direction of the e vector
    i = np.arccos(h[2]/H)
    Omega = np.arctan2(N[1]/np.linalg.norm(N[:2]),N[0]/np.linalg.norm(N[:2]))
    if (np.dot(np.cross(Nhat,evec),h))>0:
        w = np.arccos(np.dot(ehat,Nhat))
    else:
        w = -1*np.arccos(np.dot(ehat,Nhat))
    if (np.dot(np.cross(evec,r),h))>0:
        theta = np.arccos(np.dot(rhat,ehat))
    else:
        theta = -1*np.arccos(np.dot(rhat,ehat))
    
    E = np.arctan(np.tan(theta/2)*np.sqrt((1-e)/(1+e))) * 2
    M = E - e*np.sin(E)
    if i<0: # making sure no negative angles show up
        i+=2*np.pi
    if Omega<0:
        Omega+=2*np.pi
    if w<0:
        w+=2*np.pi
    if theta<0:
        theta+=2*np.pi
    if E<0:
        E+=2*np.pi
    if M<0:
        M+=2*np.pi
    return a,e,i,Omega,w,theta,E,M



#unperturbed
x_pos = data[0:,1]/1000
y_pos = data[0:,2]/1000
z_pos = data[0:,3]/1000

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_pos, y_pos, z_pos, 'r',label='orbit')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = 1700*np.cos(u)*np.sin(v)
y = 1700*np.sin(u)*np.sin(v)
z = 1700*np.cos(v)
ax.plot_wireframe(x, y, z, color="gray",label='moon')

ax.set_xlim3d(-5e3, 5e3)
ax.set_ylim3d(-5e3,5e3)
ax.set_zlim3d(-5e3,5e3)
plt.legend()
plt.show()

#perturbed

with open('perturbedtest/perturbedsattest_a2338_s10_i76.dat') as f:
    dataP = np.genfromtxt(itertools.islice(f, 0, None, 50000),delimiter=',')

x_pos_p = dataP[0:,1]/1000
y_pos_p = dataP[0:,2]/1000
z_pos_p = dataP[0:,3]/1000

#view kepler elements 
test = kepler(dataP[1][1:],u=4.9048695e12)
a = np.array([])
e = np.array([])
i = np.array([])
Omega = np.array([])
p = np.array([])
w = np.array([])

for j in range(len(dataP)):
    elements = kepler(dataP[j,1:],u=4.9048695e12)
    a = np.append(a,elements[0])
    e = np.append(e,elements[1])
    i = np.append(i,elements[2])
    p = np.append(p,elements[0]*(1-elements[1]))
    w = np.append(w,elements[4])
    Omega = np.append(Omega,elements[3])

figure,axis = plt.subplots(3,2)
figure.tight_layout()
axis[0,0].plot(dataP[:,:1]/scipy.constants.year,a/1000,label='semi-major axis')
axis[0,0].set_title('semi major axis')
axis[0,1].plot(dataP[:,:1]/scipy.constants.year,e,'r',label='eccentricity')
axis[0,1].set_title('eccentricity')
axis[1,0].plot(dataP[:,:1]/scipy.constants.year,np.degrees(i),'g',label='inclination')
axis[1,0].set_title('inclination')
axis[1,1].plot(dataP[:,:1]/scipy.constants.year,np.degrees(Omega),'orange',label='inclination')
axis[1,1].set_title('Longitude ascending node')
axis[2,0].plot(dataP[:,:1]/scipy.constants.year,p/1000,'red',label='perigee')
axis[2,0].axhline(y=1750)
axis[2,0].set_title('Perigee altitude')
axis[2,1].plot(dataP[:,:1]/scipy.constants.year,np.degrees(w),'purple',label='w')
axis[2,1].set_title('Argument of periapsis')
figure.suptitle('a2338_s10_i76')
plt.show()


#show start, mid and end orbit
orbit_data = np.genfromtxt('perturbedtest/perturbedsattest_a2338_s10_i76.dat',
                     delimiter=',',usecols=(1,2,3))

x_pos = orbit_data[0:,0]/1000
y_pos = orbit_data[0:,1]/1000
z_pos = orbit_data[0:,2]/1000

orbit_l = 4000

x_start, y_start, z_start = x_pos[:orbit_l], y_pos[:orbit_l], z_pos[:orbit_l]
x_1year, y_1year, z_1year = x_pos[(int(len(x_pos)/3-orbit_l/2)):(int(len(x_pos)/3+orbit_l/2))],\
    y_pos[int(len(y_pos)/4-orbit_l/2):int(len(y_pos)/4+orbit_l/2)],\
        z_pos[int(len(z_pos)/4-orbit_l/2):int(len(z_pos)/4+orbit_l/2)]
x_2year, y_2year, z_2year = x_pos[(int(len(x_pos)/4-orbit_l/2)):(int(len(x_pos)/4+orbit_l/2))],\
    y_pos[int(2*len(y_pos)/4-orbit_l/2):int(2*len(y_pos)/4+orbit_l/2)],\
        z_pos[int(2*len(z_pos)/3-orbit_l/2):int(2*len(z_pos)/3+orbit_l/2)]
x_3year, y_3year, z_3year = x_pos[(int(3*len(x_pos)/4-orbit_l/2)):(int(3*len(x_pos)/4+orbit_l/2))],\
    y_pos[int(3*len(y_pos)/4-orbit_l/2):int(3*len(y_pos)/4+orbit_l/2)],\
        z_pos[int(3*len(z_pos)/4-orbit_l/2):int(3*len(z_pos)/4+orbit_l/2)]
x_end, y_end, z_end = x_pos[int(len(x_pos)-orbit_l):],\
    y_pos[int(len(y_pos)-orbit_l):],\
        z_pos[int(len(z_pos)-orbit_l):]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_start, y_start, z_start, 'r',label='start orbit')
ax.plot3D(x_1year, y_1year, z_1year, 'g',label='1st year')
ax.plot3D(x_2year, y_2year, z_2year, 'b',label='2nd year')
ax.plot3D(x_3year, y_3year, z_3year, 'orange',label='3rd year')
ax.plot3D(x_end, y_end, z_end, 'purple',label='end orbit, 4th year')

u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = 1700*np.cos(u)*np.sin(v)
y = 1700*np.sin(u)*np.sin(v)
z = 1700*np.cos(v)
ax.plot_wireframe(x, y, z, color="gray",label='moon')

ax.set_xlim3d(-2.5e3, 2.5e3)
ax.set_ylim3d(-2.5e3,2.5e3)
ax.set_zlim3d(-2.5e3,2.5e3)
plt.legend()
plt.show()



