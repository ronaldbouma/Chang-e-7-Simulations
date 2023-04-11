#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 14:51:56 2022

@author: Ronald
"""
import numpy as np
import matplotlib.pyplot as plt

#estimations for bandwidth needed both internal and down to earth/central processing


def down(bandwidth,reso,N_bits,t_int,N_sat=1):
    N_bins = bandwidth/reso
    D_down = 2*N_sat*N_bins*N_bits/t_int
    
    return D_down / 1024

def datarate(bandwidth,adc,nbits):
    bandwidth_mhz = bandwidth * 1e6
    datarate = 2*bandwidth_mhz*adc /(1024**2)
    datared = 2*bandwidth_mhz*nbits /(1024**2)
    return datarate,datared

la = down(40.5,0.1,16,2,0.1)


bandwidth = np.linspace(0.5,20,100)

intersat = datarate(bandwidth, 16, 2)
downrate = down(bandwidth,0.1,2,0.1)
internal2 = datarate(bandwidth, 8, 2)[0]


fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2,sharex=True,tight_layout=True)

ax1.plot(bandwidth,intersat[1],label='intersat')
ax1.plot(bandwidth,downrate,label='down')
ax1.title.set_text('Data transfer rate')
ax1.legend()



ax2.plot(bandwidth,intersat[0],label='16 bit')
ax2.plot(bandwidth,internal2,label='8 bit')
ax2.title.set_text('on board processing')

ax2.legend()

fig.text(0.5, 0.01, 'instantaneous ' r'$\Delta \, \nu$' ' in MHz', ha='center')
fig.text(-0.02, 0.5, 'Mbit/s', va='center', rotation='vertical')
plt.show()

