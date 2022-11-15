#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:06:34 2021

@author: Ronald
"""

import numpy as np
import matplotlib.pyplot as plt

def scattering(v):
    ISM = 30/(v**(2.2))
    IPM = 100/(v**(2))
    return ISM,IPM

def limit(v,r):
    c = 299792458
    wavelength = c/(v*1e6)
    difr = 1.22*wavelength/(r*1e3)
    return np.degrees(difr) *60

def confusion(res,v):
    return 16*(res)**(1.54)*(v/74)**(-0.7)

def time(v,theta,N,eta,b):
    return 3.3 * (N/100)**(-2) * 1/(eta**2) * (b/0.1)**(-1) * (v)**(-0.66) * (theta)**(-3.08)

def time2(s,v,T,dv,N):
    return 44/N**2 * (s)**(-2) * (T/10)**2 * (dv/0.1)**(-1) * (v)**4

freq_range = np.linspace(2,50,300)

ISM,IPM = scattering(freq_range)

limit_radius = limit(freq_range,1700)
limit_diameter = limit(freq_range,3400)

plt.plot(freq_range,np.log10(ISM),label='ISM limit')
plt.plot(freq_range,np.log10(IPM),label='IPM limit')
plt.plot(freq_range,np.log10(limit_radius),label='Diffraction limit (d=1700km)')
plt.xlabel('frequency in MHz')
plt.ylabel('Log of limit in arcminutes')
plt.legend()
plt.show()

S_confusion = confusion(limit_radius,freq_range)

plt.plot(freq_range,S_confusion,label='confusion limit')
plt.xlabel('Frequency in MHz')
plt.ylabel('Confusion limit in mJy')

survey = time(freq_range,limit_radius,6,1,0.1)
survey2 = time2(0.1,freq_range,16.7,0.1,3)
survey3 = time2(1,freq_range,16.7,0.1,3)

plt.plot(freq_range,survey,label='to confusion limit')
plt.plot(freq_range,survey2,label='to 100 mJy')
plt.plot(freq_range,survey3,label='to 1 Jy')
plt.xlabel('Frequency in MHz')
plt.ylabel('survey time in days for one sterradian')
plt.axis([0,15,0,10000])
plt.legend()
plt.show()
