#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:35:05 2022

@author: Ronald
"""

#cd
import os
import sys

#packages
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import functions2 as funcs



def load_map(freq,B, highres=False, cl=False):
    path = os.path.expanduser('~/desktop/thesis/output/')
    if highres == False:
        if cl==False:
            map_load = np.load('{}F{}MHz_B{}_notcleaned.npz'.format(path,freq,B))
        elif cl==True:
            map_load = np.load('{}cleaned/F{}MHz_B{}.npz'.format(path,freq,B))
    elif highres==True:
        map_load = np.load('{}hr/hr_F{}MHz_B{}_notcleaned.npz'.format(path,freq,B))
    
    if cl == True:
        return map_load['full_sky']
    elif cl==False:
        return map_load['reconstruction']


    
       
freq = 10
res = True
B = np.array([100000,500000,1000000])

models = funcs.make_model(2**8, freq*1e6)[0]
maps= abs(load_map(freq,B[0],highres=res,cl=False)), abs(load_map(freq,B[1],highres=res,cl=False)), abs(load_map(freq,B[2],highres=res,cl=False))

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
plt.axes(ax1)
hp.mollview(np.log10(models),hold=True,cbar=False)


plt.axes(ax2)
hp.mollview(np.log10(maps[0]),hold=True,cbar=False)



plt.axes(ax3)
hp.mollview(np.log10(maps[1]),hold=True,cbar=False)


plt.axes(ax4)
hp.mollview(np.log10(maps[2]),hold=True,cbar=False)


plt.show()


##cleaned

freq = 25
cl = True
B = np.array([5000,10000,50000])

models =funcs.make_model(2**8, freq*1e6)[0]
full_sky = load_map(freq,B[2],cl=cl)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
plt.axes(ax1)
hp.mollview((abs(models)),hold=True,cbar=False)

plt.axes(ax2)
hp.mollview((abs(full_sky[0])),hold=True,cbar=False)


plt.axes(ax3)
hp.mollview((abs(full_sky[5])),hold=True,cbar=False)


plt.axes(ax4)
hp.mollview((abs(full_sky[-1])),hold=True,cbar=False)

plt.show()






