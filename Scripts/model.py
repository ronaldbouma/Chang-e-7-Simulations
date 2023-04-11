#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 09:50:20 2022

@author: Ronald
"""

#cd
import os
import sys


#packages
import numpy as np
import functions2 as funcs
import healpy as hp
import matplotlib.pyplot as plt

#script to project the model with alterations to the model

##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

frequency = 10e6
freq2 = 50e6

NSIDE = 2**8

NPIX = hp.nside2npix(NSIDE)
##location of fossil
direction = np.radians([[110,70],[130,60],[110,40]])
size_ang = np.radians([2,2,2])
intensity_base = np.array([75,150,300])
intensities_multiplier = (freq2/10e6)**(-0.5)

lines = np.radians([105,135]),np.radians([35,75])
theta_b = [100,140]
phi_b = [30,80]

v_line_1 = np.radians([np.linspace(theta_b[0],theta_b[1],100), [phi_b[0]]*100])
v_line_2 = np.radians([np.linspace(theta_b[0],theta_b[1],100), [phi_b[1]]*100])
h_line_1 = np.radians([[[theta_b[0]]*100],[np.linspace(phi_b[0],phi_b[1],100)]] )
h_line_2 = np.radians([[[theta_b[1]]*100],[np.linspace(phi_b[0],phi_b[1],100)]] )

intensities_fossil = intensity_base
model_I, model_x = funcs.make_model(NSIDE, frequency, object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)

intensities_fossil = intensity_base * intensities_multiplier
model_I2, model_x = funcs.make_model(NSIDE, frequency*5, object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)


fig = plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.subplots_adjust(hspace=0.2)

hp.mollview(np.log(model_I),title='Log view model at 10MHz',cbar=True,notext=False,hold=True)
#hp.projplot(np.radians(120),np.radians(60),'o')
#hp.projplot(lines,'ro-')
hp.projplot(v_line_1[0],v_line_1[1],',')
hp.projplot(v_line_2[0],v_line_2[1],',')
hp.projplot(h_line_1[0],h_line_1[1],',')
hp.projplot(h_line_2[0],h_line_2[1],',')
#hp.graticule(dpar=45)

plt.subplot(2,1,2)

hp.mollview(np.log(model_I2),title='Log view model at 50MHz',cbar=True,notext=False,hold=True)
hp.projplot(v_line_1[0],v_line_1[1],',')
hp.projplot(v_line_2[0],v_line_2[1],',')
hp.projplot(h_line_1[0],h_line_1[1],',')
hp.projplot(h_line_2[0],h_line_2[1],',')
#hp.graticule(dpar=45)


plt.tight_layout()

plt.show()




