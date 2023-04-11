#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 11:22:03 2022

@author: Ronald
"""

#script to compare several maps, in order to inspect the usefulness 
#of adding more meaurements

#cd
import os
direct = os.getcwd()
if direct != '/Users/Ronald/Desktop/Thesis/Scripts':
    os.chdir('desktop/thesis/scripts')

#packages
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import functions2 as funcs


freq = 25

path = '~/desktop/thesis/output/'

path_1 = '{}F{}MHz_B100000_notcleaned.npz'.format(path,freq)
path_2 = '{}F{}MHz_B500000_notcleaned.npz'.format(path,freq)
path_3 = '{}F{}MHz_B1000000_notcleaned.npz'.format(path,freq)

rec_1 = np.load(os.path.expanduser(path_1),allow_pickle=True)['reconstruction']
rec_2 = np.load(os.path.expanduser(path_2),allow_pickle=True)['reconstruction']
rec_3 = np.load(os.path.expanduser(path_3),allow_pickle=True)['reconstruction']

NSIDE = 2**8
frequency = freq*1e6
NPIX = hp.nside2npix(NSIDE)
##location of fossil
direction = np.radians([[110,70],[130,60],[110,40]])
size_ang = np.radians([2,2,2])
intensity_base = np.array([75,150,300])
intensities_multiplier = (frequency/10e6)**(-0.5)
intensities_fossil = intensity_base * intensities_multiplier
model_I, model_x = funcs.make_model(NSIDE, frequency, object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)



fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2) 

fig.suptitle=('Noise reduction at {}MHz'.format(freq))


plt.axes(ax1)
hp.mollview(np.log10(abs(model_I)),cbar=False,title='input model', hold=True)

plt.axes(ax2)
hp.mollview(np.log10(abs(rec_1)),cbar=False,title='100.000 baselines', hold=True)

plt.axes(ax3)
hp.mollview(np.log10(abs(rec_2)),cbar=False,title='500.000 baselines', hold=True)

plt.axes(ax4)
hp.mollview(np.log10(abs(rec_3)),cbar=False,title='1.000.000 baselines', hold=True)

plt.show()



