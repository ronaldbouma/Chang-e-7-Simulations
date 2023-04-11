#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:36:06 2022

@author: Ronald
"""

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


#noise analysis for maps that have been cleaned

def spectra(noisemap,mask=None,size=400):
    if type(mask) != type(None):
        masked_pix = hp.query_strip(hp.npix2nside(len(noisemap)),np.radians(mask[0]),np.radians(mask[1]))
        noisemap[masked_pix] = 0
    newmap = hp.pixelfunc.remove_monopole(noisemap)
    spectrum = hp.sphtfunc.anafast(newmap,lmax=size)[1:]
    return spectrum

def spectopower(spectrum):
    power = np.zeros(len(spectrum))
    for i in range(len(spectrum)):
        #power[i] = spectrum[i] * (2*i + 1)
        power[i] = spectrum[i] * i*(i+1)
    return power

def residuals(freq,B):
    path = os.path.expanduser('~/desktop/thesis/output/')
    map_load = np.load('{}cleaned/F{}MHz_B{}000.npz'.format(path,freq,B))
    map_cleaned = map_load['full_sky'][-1]
    model = funcs.make_model(2**7, freq*1e6)[0]
    pixels = map_load['pixels']
    model_active = np.copy(model)
    for i in range(len(pixels)):
        pixels_active = pixels[i].astype(int)
        model_active[pixels_active] = model_active[pixels_active] * 0.8
    resi = abs(abs(model_active)-abs(map_cleaned))
    return resi, model_active

def noise_cleaned(freq):
    path = os.path.expanduser('~/desktop/thesis/output/')
    map_load = np.load('{}cleaned/F{}MHz_B50000.npz'.format(path,freq))
    full_sky = map_load['full_sky']
    model = funcs.make_model(2**7, freq*1e6)[0]
    model_active = np.copy(model)
    pixels = map_load['pixels']
    residuals = np.zeros(np.shape(full_sky))
    models = np.zeros(np.shape(full_sky))
    model_active = np.copy(model)
    for i in range(len(pixels)):
        models[i] = model_active
        pixels_active = pixels[i].astype(int)
        model_active[pixels_active] = model_active[pixels_active] * 0.8
        residuals[i] = abs(abs(model_active)-abs(full_sky[i]))
    return residuals, full_sky

freqs = [10,15,20,25,30,50]
l_length = 500
mask = np.array((0,91))
resi_10 = np.zeros((2,hp.nside2npix(2**7)))
residual = np.zeros((len(freqs)-1,3,hp.nside2npix(2**7)))
spectras_10 = np.zeros((2,l_length))
power_10 = np.zeros((2,l_length))
spectras = np.zeros((len(freqs)-1,3,l_length))
power = np.zeros((len(freqs)-1,3,l_length))
models_cleaned = np.zeros((len(freqs),3,hp.nside2npix(2**7)))

for i in tqdm(range(len(freqs))):
    if i ==0:
        resi_10[0], models_cleaned[0][0] = residuals(10,5)
        resi_10[1], models_cleaned[0][1] = residuals(10,10)
        spectras_10[0]  = spectra(resi_10[0],mask=mask,size=l_length)
        spectras_10[1] = spectra(resi_10[1],mask=mask,size=l_length)
        power_10[0] = spectopower(spectras_10[0])
        power_10[1] = spectopower(spectras_10[1])
    else:
        residual[i-1][0],models_cleaned[i][0] = residuals(freqs[i],5)
        residual[i-1][1],models_cleaned[i][1] = residuals(freqs[i],10)
        residual[i-1][2],models_cleaned[i][2] = residuals(freqs[i],50)
        spectras[i-1][0] = spectra(residual[i-1][0],mask=mask,size=l_length)
        spectras[i-1][1] = spectra(residual[i-1][1],mask=mask,size=l_length)
        spectras[i-1][2] = spectra(residual[i-1][2],mask=mask,size=l_length)
        power[i-1][0] = spectopower(spectras[i-1][0])
        power[i-1][1] = spectopower(spectras[i-1][1])
        power[i-1][2] = spectopower(spectras[i-1][2])
        
fig, ax = plt.subplots(nrows=3,ncols=2,tight_layout=True)
ax[0][0].plot(np.log(power_10[0]),color='y',label='_5.0000')
ax[0][0].plot(np.log(power_10[1]),color='b',label='_10.000')
ax[0][0].set_title('10MHz')

ax[0][1].plot(np.log(power[0][0]),color='y')
ax[0][1].plot(np.log(power[0][1]),color='b')
ax[0][1].plot(np.log(power[0][2]),color='g')
ax[0][1].set_title('15MHz')

ax[1][0].plot(np.log(power[1][0]),color='y',label='5.000')
ax[1][0].plot(np.log(power[1][1]),color='b',label='10.000')
ax[1][0].plot(np.log(power[1][2]),color='g',label='50.000')
ax[1][0].set_title('20MHz')

ax[1][1].plot(np.log(power[2][0]),color='y',label='5.000')
ax[1][1].plot(np.log(power[2][1]),color='b',label='10.000')
ax[1][1].plot(np.log(power[2][2]),color='m',label='50.000')
ax[1][1].set_title('25MHz')
ax[1][1].legend(bbox_to_anchor =(1.02, 1.0))

ax[2][0].plot(np.log(power[3][0]),color='y',label='5.000')
ax[2][0].plot(np.log(power[3][1]),color='b',label='10.000')
ax[2][0].plot(np.log(power[3][2]),color='m',label='50.000')
ax[2][0].set_title('30MHz')

ax[2][1].plot(np.log(power[4][0]),color='y')
ax[2][1].plot(np.log(power[4][1]),color='b')
ax[2][1].plot(np.log(power[4][2]),color='m')
ax[2][1].set_title('50MHz')

fig.text(0.5, 0.01, '$\ell$', ha='center')
fig.text(0.00, 0.5, '$Log_{10}\, \ell\,*(\ell+1)\, C_\ell$', va='center', rotation='vertical')
fig.suptitle('angular spectral noise of cleaned maps')
plt.show()


##noise average
pix = hp.pixelfunc.nside2npix(2**5)
noise_5 = np.zeros((5,pix))
noise_10 = np.zeros((5,pix))
noise_50 = np.zeros((5,pix))
mean_noise_5 = np.zeros(5)
mean_noise_10 =np.zeros(5)
mean_noise_50 = np.zeros(5)
mean_noise_10mhz = np.array((hp.pixelfunc.ud_grade(resi_10[0],2**5), hp.pixelfunc.ud_grade(resi_10[1],2**5)))
mean_noise_10mhz[mean_noise_10mhz==0] = np.nan
noise_10mhz = np.nanmean(mean_noise_10mhz,axis=1)

for j in range(len(residual)):
    temp_5 = hp.pixelfunc.ud_grade(residual[j][0],2**5) 
    temp_10 = hp.pixelfunc.ud_grade(residual[j][1],2**5) 
    temp_50 =hp.pixelfunc.ud_grade(residual[j][2],2**5) 
    temp_5[temp_5==0] = np.nan
    temp_10[temp_10==0] = np.nan
    temp_50[temp_50==0] = np.nan
    noise_5[j] = temp_5
    noise_10[j] = temp_10
    noise_50[j] = temp_50
    mean_noise_5[j] = np.nanmean(temp_5)
    mean_noise_10[j] = np.nanmean(temp_10)
    mean_noise_50[j] = np.nanmean(temp_50)
    
width = 1/5-(0.1/5)
labels = ['15MHz','20MHz','25MHz','30MHz']
x=np.arange(4)


fig,ax = plt.subplots()
rects1 = ax.bar(x-width*2,mean_noise_5[:4],width,label='5.000')
rects2 = ax.bar(x-width,mean_noise_5[:4]*(1/np.sqrt(2)),width,label='5.000 / \u221A2')
rects3 = ax.bar(x,mean_noise_10[:4],width,label='10.000')
rects4 = ax.bar(x+width,mean_noise_5[:4]*(1/np.sqrt(10)),width,label='5.000 / \u221A10')
rects5 = ax.bar(x+width*2,mean_noise_50[:4],width,label='50.000')

ax.set_ylabel('Average noise in Jansky')
ax.set_title('Noise comparison')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.show()








##comparison different cleaning stages

frequen = 30
mapz,full_sky = noise_cleaned(frequen)
mask = np.array([0,91])
l_size = 500
spectras = np.zeros((len(mapz),l_size))
power = np.zeros(np.shape(spectras))
select = int(len(power)/2)
last = len(mapz)

for k in range(len(mapz)):
    spectras[k] = spectra(mapz[k],mask=mask,size=l_size)
    power[k] = spectopower(spectras[k])

plt.plot(np.log10(power[0]),label='uncleaned')
plt.plot(np.log10(power[select]),label='cleaned {} times'.format(select))
plt.plot(np.log10(power[-1]),label='cleaned {} times'.format(last))
plt.xlabel('order of $\ell$')
plt.ylabel('$Log_{10}\, \ell\,*(\ell+1)\, C_\ell$')
plt.title('Noise angular spectrum change by CLEAN routine, {}MHz'.format(frequen))
plt.legend()
plt.show()

fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(15,15))

plt.axes(ax1)
hp.mollview((abs(full_sky[0])),hold=True,cbar=False,title='Not cleaned, linear scale')
plt.axes(ax2)
hp.mollview(np.log10((abs(full_sky[0]))),hold=True,cbar=False,title='Not cleaned, log scale')
plt.axes(ax3)
hp.mollview((abs(full_sky[select])),hold=True,cbar=False,title='Cleaned {} times, linear scale'.format(select))

plt.axes(ax4)
hp.mollview((np.log10(abs(full_sky[select]))),hold=True,cbar=False,title='Cleaned {} times, linear scale'.format(select))

plt.axes(ax5)
hp.mollview((abs(full_sky[-1])),hold=True,cbar=False,title='Cleaned {} times, linear scale'.format(last))

plt.axes(ax6)
hp.mollview(np.log10((abs(full_sky[-1]))),hold=True,cbar=False,title='Cleaned {} times, log scale'.format(last))

plt.show()






