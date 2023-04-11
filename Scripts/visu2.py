#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:05:41 2023

@author: Ronald
"""

#visualization gives sep[erate gnomview on target plus mollview.\
    #inputs are command line based

#cd
import os
import sys
os.chdir(os.path.expanduser('~/desktop/thesis/scripts'))
#packages
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp



frequency = 30
baselines = 1000000
baselines_thousands = baselines
cleaned = False
high_res = False

coords = [60,-40,0]

if cleaned == True:
    path = '~/desktop/thesis/output/cleaned/F{}MHz_B{}.npz'.format(frequency,baselines)
elif high_res == True:
    path = '~/desktop/thesis/output/hr/hr_F{}MHz_B{}_notcleaned.npz'.format(frequency,baselines)
else:
    path = '~/desktop/thesis/output/F{}MHz_B{}_notcleaned.npz'.format(frequency,baselines)


if cleaned==False:
    sim = np.load(os.path.expanduser(path),allow_pickle=True)
    rec = sim['reconstruction']
    
    fig,(ax1,ax2)=plt.subplots(ncols=2,gridspec_kw={'width_ratios': [1.7, 1]},figsize=(12,8))
    plt.axes(ax1)
    pos = ax1.get_position()
    new_pos = [pos.x0, pos.y0+0.05, pos.width, pos.height]
    ax1.set_position(new_pos)
    hp.mollview(np.log10(abs(rec)),hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
    hp.visufunc.graticule()
    hp.visufunc.projplot(coords[0],coords[1],'ro',mfc='none',lonlat=True)
    plt.axes(ax2)
    hp.gnomview(np.log10(abs(rec)),rot=coords,xsize=300,hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
    #plt.tight_layout()
    plt.show()
    
if cleaned == True:
    sim = np.load(os.path.expanduser(path))
    full_sky = sim['full_sky']
    
    fig,(ax1,ax2)=plt.subplots(ncols=2,gridspec_kw={'width_ratios': [1.7, 1]},figsize=(12,8))
    plt.axes(ax1)
    pos = ax1.get_position()
    new_pos = [pos.x0, pos.y0+0.05, pos.width, pos.height]
    ax1.set_position(new_pos)
    hp.mollview(np.log10(abs(full_sky[-1])),hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
    hp.visufunc.graticule()
    hp.visufunc.projplot(coords[0],coords[1],'rx',lonlat=True)
    plt.axes(ax2)
    hp.gnomview(np.log10(abs(full_sky[-1])),rot=coords,xsize=300,hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
    #plt.tight_layout()
    plt.show()
    
    
    
    
    