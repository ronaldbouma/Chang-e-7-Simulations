#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:49:40 2022

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

#script to produce the maps from the command line
#command line arguments: frequency (in mhz), baseline amount in terms of thousands, cleaned or high res
#output is a mollzoom function (combined clickable mollview and gnomview)
#arguments
arguments = len(sys.argv)
frequency = sys.argv[1]
baselines = 1000* int(sys.argv[2])
baselines_thousands = int(sys.argv[2])
if baselines_thousands == 1000:
    baselines_thousands = '1.000'

cleaned = False
high_res = False
gnom_select = False

if arguments ==4:
    if sys.argv[3] == 'cl':
        cleaned = True
    elif sys.argv[3] == 'hr':
        high_res = True

coord_option=False
coords = [0,0,0]
if arguments ==6:
    coords[0] = sys.argv[4]
    coords[1] = sys.argv[5]
    coord_option=True
    

if cleaned == True:
    path = '~/desktop/thesis/output/cleaned/F{}MHz_B{}.npz'.format(frequency,baselines)
elif high_res == True:
    path = '~/desktop/thesis/output/hr/hr_F{}MHz_B{}_notcleaned.npz'.format(frequency,baselines)
else:
    path = '~/desktop/thesis/output/F{}MHz_B{}_notcleaned.npz'.format(frequency,baselines)

if cleaned==False:
    sim = np.load(os.path.expanduser(path),allow_pickle=True)
    rec = sim['reconstruction']

    hp.mollzoom(np.log10(abs(rec)), title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
    
    hp.mollzoom(abs(rec),xsize=300,title='{} Mhz, {}.000 baselines'.format(frequency,baselines_thousands))
    #degraded_map = hp.pixelfunc.ud_grade(rec,2**5)
    #hp.mollzoom(np.log10(abs(degraded_map)),xsize=300,title='{} Mhz, {}.000 baselines degraded map'.format(frequency,baselines_thousands))
    plt.show()

    if coord_option == True:
        fig,(ax1,ax2)=plt.subplots(ncols=2,gridspec_kw={'width_ratios': [1.7, 1]},figsize=(12,8))
        plt.axes(ax1)
        pos = ax1.get_position()
        new_pos = [pos.x0, pos.y0+0.05, pos.width, pos.height]
        ax1.set_position(new_pos)
        hp.mollview(np.log10(abs(rec)),hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
        hp.visufunc.graticule()
        hp.visufunc.projplot(coords[0],coords[1],'rx',lonlat=True)
        plt.axes(ax2)
        hp.gnomview(np.log10(abs(rec)),rot=coords,hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
        #plt.tight_layout()
        plt.show()

if cleaned == True:
    sim = np.load(os.path.expanduser(path))
    full_sky = sim['full_sky']
    cycles = len(full_sky)
    hp.mollview(abs(full_sky[0]),title='')
    hp.mollview(abs(full_sky[-1]),title='')
    hp.mollzoom(abs(full_sky[-1]),title='Cleaned map {}MHz and {}.000 Baselines, linear scale'.format(frequency,baselines_thousands))
    hp.mollzoom(np.log10(abs(full_sky[-1])),title='Cleaned map {}MHz and {}.000 Baselines, log scale'.format(frequency, baselines_thousands))
    plt.show()
    
    if coord_option==True:
        fig,(ax1,ax2)=plt.subplots(ncols=2,gridspec_kw={'width_ratios': [1.7, 1]},figsize=(12,8))
        plt.axes(ax1)
        pos = ax1.get_position()
        new_pos = [pos.x0, pos.y0+0.05, pos.width, pos.height]
        ax1.set_position(new_pos)
        hp.mollview(np.log10(abs(full_sky[-1])),hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
        hp.visufunc.graticule()
        hp.visufunc.projplot(coords[0],coords[1],'rx',lonlat=True)
        plt.axes(ax2)
        hp.gnomview(np.log10(abs(full_sky[-1])),rot=coords,hold=True, title='{} Mhz, {}.000 baselines, log scale'.format(frequency,baselines_thousands))
        #plt.tight_layout()
        plt.show()






