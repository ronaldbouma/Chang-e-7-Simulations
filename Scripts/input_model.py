#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:44:49 2022

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

os.chdir(os.path.expanduser('~/desktop/thesis/images'))

frequencies = np.array([10,15,20,15,30,50])

#make model

NSIDE = 2**7
NPIX = hp.nside2npix(NSIDE)
##location of fossil
direction = np.radians([[110,70],[130,60],[110,40]])
size_ang = np.radians([2,2,2])
intensity_base = np.array([250,500,1000])

for i in range(len(frequencies)):
    intensities_multiplier = (frequencies[i]/10)**(-0.5)
    intensities_fossil = intensity_base * intensities_multiplier
    model = funcs.make_model(NSIDE, frequencies[i]*1e6,object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)[0]

    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    plt.axes(ax1)
    hp.mollview(model, title='input model {} Mhz'.format(frequencies[i]), hold=True)

    plt.axes(ax2)
    hp.mollview(np.log(model),title='input model {} Mhz, log scale'.format(frequencies[i]), hold=True)
    
    fig.savefig('model_{}MHz'.format(frequencies[i]),dpi=300)