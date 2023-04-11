#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:48:50 2023

@author: Ronald
"""

#old visualization, incorperated into visufunc.

#cd
import os
direct = os.getcwd()
if direct != '/Users/Ronald/Desktop/Thesis/Scripts':
    os.chdir('desktop/thesis/scripts')
import sys

#packages
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import functions2 as funcs
from tqdm import tqdm


freq = 20
wavelength = 299792458/(freq*1e6)
NSIDE = 2**7
NPIX = hp.pixelfunc.nside2npix(NSIDE)
number = 50000

path = os.path.expanduser('~/desktop/thesis/output/')
map_load = np.load('{}cleaned/F{}MHz_B{}.npz'.format(path,freq,number))


model_x = funcs.make_model(NSIDE, freq*1e6)[1]
cleaned = map_load['Vs_cleaning']
cleaned_tot=map_load['Vs_clean_tot']
Positions = map_load['Positions']
full_sky = map_load['full_sky']

image = funcs.reconstruct(Positions,cleaned_tot,NPIX,model_x,wavelength)
hp.mollview(abs(image),cbar=False,title='Map of removed visibilities')
hp.mollview(np.log10(abs(image)))

hp.mollview(abs(full_sky[0]))
hp.mollview(abs(full_sky[-1]))

image_sub = abs(full_sky[-1]-full_sky[0])

hp.mollview((image_sub),cbar=False,title='clean image')
hp.mollview(np.log10(image_sub),cbar=False,title='clean image log view')

#


