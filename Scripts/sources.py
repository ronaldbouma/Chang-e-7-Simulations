#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:43:05 2022

@author: Ronald
"""

#estimation of sources visible given the equation from Jester et al.\
    #needs simulation results added in mean noise, very hands on script.

import numpy as np
import matplotlib.pyplot as plt

def Nsources(S,v):
    N = 1800* (S/0.01)**(-1.3) * (v/10)**(-0.7)
    
    
    
    return N

freqs = np.array([10,15,20,25,30])

Srange = np.logspace(-1,2,100)
N_sources = np.zeros((len(freqs),len(Srange)))
for i in range(len(freqs)):
    N_sources[i] = Nsources(Srange,freqs[i])

mean_noise = np.array([[1.52734023, 0.6238406 , 0.4162006 ],
       [0.40696378, 0.19719819, 0.14090664],
       [0.21131973, 0.12520145, 0.13861465],
       [0.1496512 , 0.09283062, 0.07200494],
       [0.12410382, 0.07685074, 0.05778135]])
mean = np.log10(mean_noise[:,0])
numbers = Nsources(10**mean,freqs)
maximum = np.max(N_sources)

plt.plot(np.log10(Srange),(N_sources[0]),color='b',label='{}MHz'.format(freqs[0]))
plt.plot(np.log10(Srange),(N_sources[1]),color='y',label='{}MHz'.format(freqs[1]))
plt.plot(np.log10(Srange),(N_sources[2]),color='g',label='{}MHz'.format(freqs[2]))
plt.plot(np.log10(Srange),(N_sources[3]),color='r',label='{}MHz'.format(freqs[3]))
plt.plot(np.log10(Srange),(N_sources[4]),color='m',label='{}MHz'.format(freqs[4]))
plt.vlines(mean[0],0,numbers[0],color='b',ls='--')
plt.vlines(mean[1],0,numbers[1],color='y',ls='--')
plt.vlines(mean[2],0,numbers[2],color='g',ls='--')
plt.vlines(mean[3],0,numbers[3],color='r',ls='--')
plt.vlines(mean[4],0,numbers[4],color='m',ls='--')
plt.hlines(numbers[0],-1,mean[0],ls='--',color='b')
plt.hlines(numbers[1],-1,mean[1],ls='--',color='y')
plt.hlines(numbers[2],-1,mean[2],ls='--',color='g')
plt.hlines(numbers[3],-1,mean[3],ls='--',color='r')
plt.hlines(numbers[4],-1,mean[4],ls='--',color='m')
plt.xlabel('Intensity of the sources in Jansky (Log scale)')
plt.ylabel('Sources per square degree')
plt.xlim(0.5,-1)
plt.ylim(0,60)
plt.gca().invert_xaxis()
plt.title('Estimation of sources visible after 4 years')
plt.legend(loc='upper right')
plt.show()







