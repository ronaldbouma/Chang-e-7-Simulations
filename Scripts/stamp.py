#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:32:48 2022

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
import matplotlib.pyplot as plt


##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

NSIDE = 2**7
frequency = 15*1e6

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

#time selection
max_years = 2
t_int = 0.1
time_span = [0,max_years*31556926]

baseline_amount = 1000
target = (110,40)


t_points = funcs.select_visible(time_span, np.radians(target), int(baseline_amount), sat1_state, sat2_state, ground_state)
Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
Positions = Ds[3:6]
orientations = Ds[6:]

size = np.array((1,1))/120


stamp, degree_grid = funcs.grid_testing(target, size, Positions, wavelength,resolution=(300,300),source_amount=1,)

degree_grid = (degree_grid[0]-target[0]) *3600,(degree_grid[1] -target[1]) *3600


uvw,uv,psf,grid,points = funcs.uvw_transform(target, Positions, wavelength,grid_resolution=300,psf_output=True)

lm = 1/(np.mean(np.diff(points[0]))), 1/abs(np.mean(np.diff(points[1])))

plt.figure(1)
plt.scatter(uv[0],uv[1],s=1)
plt.title('')

plt.figure(2)
plt.imshow(grid,origin='lower',extent=(points[0][0],points[0][-1],points[1][0],points[1][-1]))
plt.xlabel('u')
plt.ylabel('v')
plt.title('uv coverage')

plt.figure(3)
#plt.imshow(psf,origin='lower', extent=(lm[0]*3600,lm[1]*3600,lm[0]*3600,lm[1]*3600))
plt.imshow(psf,extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
plt.xlim(4,-4)
plt.ylim(4,-4)
#plt.axis('off')
plt.title('PSF')

plt.figure(4)
plt.imshow(abs(stamp),extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.title('Image')
plt.xlabel('arcsec')
plt.ylabel('arcsec')
plt.show()



fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(grid,origin='lower',extent=(points[0][0]/1e3,points[0][-1]/1e3,points[1][0]/1e3,points[1][-1]/1e3))
plt.xlabel('u * 1e3')
plt.ylabel('v * 1e3')
plt.title('uv coverage')


plt.subplot(1,2,2)
#plt.imshow(psf,origin='lower', extent=(lm[0]*3600,lm[1]*3600,lm[0]*3600,lm[1]*3600))
plt.imshow(psf,extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
#plt.gca().invert_yaxis()
#plt.gca().invert_xaxis()
plt.xlim(4,-4)
plt.ylim(4,-4)
#plt.axis('off')
plt.title('PSF')

plt.show()

#Positions1 = Positions[:,:baseline_amount[0],:]
#Positions2 = Positions[:,:baseline_amount[1],:]
#Positions3 = Positions[:,:baseline_amount[2],:]


#test1,source_map, degree_grid = funcs.grid_testing(target, (1,1), Positions1, wavelength, resolution=(400,400), source_amount=10, seed=123456)
#test2,source_map, degree_grid = funcs.grid_testing(target, (1,1), Positions2, wavelength, resolution=(400,400), source_amount=10, seed=123456)
#test3,source_map, degree_grid = funcs.grid_testing(target, (1,1), Positions3, wavelength, resolution=(400,400), source_amount=10, seed=123456)

#plt.imshow(abs(source_map),extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')


#plt.imshow(abs(test1),extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
#plt.imshow(abs(test2),extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
#plt.imshow(abs(test3),extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')


#plt.show()


#multi uv map
grid_resolution =300
targets = np.array([[45,60],[90,0],[90,150],[150,60],[150,120]])
u_array = np.zeros([len(targets),3*baseline_amount])
v_array = np.zeros([len(targets),3*baseline_amount])
psf_array = np.zeros([len(targets),grid_resolution,grid_resolution])
grid_array = np.zeros([len(targets),grid_resolution,grid_resolution])
points_array = np.zeros([len(targets),2,grid_resolution])

for i in range(len(targets)):
    uvw,uv,psf,grid,points = funcs.uvw_transform(targets[i], Positions, wavelength,grid_resolution=grid_resolution,psf_output=True)
    u_array[i] = uv[0]
    v_array[i] = uv[1]
    psf_array[i] = psf
    grid_array[i] = grid
    points_array[i] = points
    
    


fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8),(ax9,ax10)) = plt.subplots(nrows=len(targets),ncols=2,sharex=False, sharey=False,figsize=(8,15))

plt.axes(ax1)

plt.imshow(grid_array[0],origin='lower',extent=(points_array[0][0][0]/1e3,points_array[0][0][-1]/1e3,points_array[0][1][0]/1e3,points_array[0][1][-1]/1e3))

plt.title('uv coverage, 45, 30')

plt.axes(ax2)

plt.imshow(psf_array[0],extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.xlim(4,-4)
plt.ylim(4,-4)
plt.title('PSF, 45, 30')



plt.axes(ax3)
plt.imshow(grid_array[1],origin='lower',extent=(points_array[1][0][0]/1e3,points_array[1][0][-1]/1e3,points_array[1][1][0]/1e3,points_array[1][1][-1]/1e3))

plt.title('uv coverage, 0, 0')


plt.axes(ax4)

plt.imshow(psf_array[1],extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.xlim(4,-4)
plt.ylim(4,-4)
plt.title('PSF, 0, 0')


plt.axes(ax5)
plt.imshow(grid_array[2],origin='lower',extent=(points_array[2][0][0]/1e3,points_array[2][0][-1]/1e3,points_array[2][1][0]/1e3,points_array[2][1][-1]/1e3))
plt.title('uv coverage, 0, 40')
plt.ylabel('v / 1000')

plt.axes(ax6)

plt.imshow(psf_array[2],extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.xlim(4,-4)
plt.ylim(4,-4)
plt.title('PSF, 0, 40')
plt.ylabel('arcsec')

plt.axes(ax7)
plt.imshow(grid_array[3],origin='lower',extent=(points_array[3][0][0]/1e3,points_array[3][0][-1]/1e3,points_array[3][1][0]/1e3,points_array[3][1][-1]/1e3))
plt.title('uv coverage, -60, 60')


plt.axes(ax8)

plt.imshow(psf_array[3],extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.xlim(4,-4)
plt.ylim(4,-4)
plt.title('PSF, -60, 60')


plt.axes(ax9)
plt.imshow(grid_array[4],origin='lower',extent=(points_array[4][0][0]/1e3,points_array[4][0][-1]/1e3,points_array[4][1][0]/1e3,points_array[4][1][-1]/1e3))
plt.title('uv coverage, -60, 120')
plt.xlabel('u / 1000')

plt.axes(ax10)

plt.imshow(psf_array[1],extent=(degree_grid[1][0],degree_grid[1][-1],degree_grid[0][0],degree_grid[0][-1]),origin='lower')
plt.xlim(4,-4)
plt.ylim(4,-4)
plt.title('PSF, -60, 120')
plt.xlabel('arcsec')
plt.tight_layout()

plt.show()




