#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:17:08 2022

@author: Ronald
"""
#noise analysis for use with reconstructed maps
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
from tqdm import tqdm


#frequency maps
freqs = [10,15,20,25,30]

#function to load maps
def noise_map(freq,degradeNSIDE=None,mask=None,selecttwo=False):
    path = '~/desktop/thesis/output/'
    
    if freq == 10:
        path_1 = '{}hr/hr_F{}MHz_B100000_notcleaned.npz'.format(path,freq)
        path_2 = '{}hr/hr_F{}MHz_B500000_notcleaned.npz'.format(path,freq)
        path_3 = '{}hr/hr_F{}MHz_B1000000_notcleaned.npz'.format(path,freq)
    elif selecttwo==True:
        path_1 = '{}hr/hr_F{}MHz_B100000_notcleaned.npz'.format(path,freq)
        path_2 = '{}hr/hr_F{}MHz_B500000_notcleaned.npz'.format(path,freq)
    else:
        path_1 = '{}F{}MHz_B100000_notcleaned.npz'.format(path,freq)
        path_2 = '{}F{}MHz_B500000_notcleaned.npz'.format(path,freq)
        path_3 = '{}F{}MHz_B1000000_notcleaned.npz'.format(path,freq)
    
    
    if freq == 10:
        NSIDE = 2**8
    elif selecttwo==True:
        NSIDE = 2**8
    else:
        NSIDE = 2**7
    
    if freq==10:
        rec_1 = np.load(os.path.expanduser(path_1),allow_pickle=True)['reconstruction']
        rec_2 = np.load(os.path.expanduser(path_2),allow_pickle=True)['reconstruction']
        rec_3 = np.load(os.path.expanduser(path_3),allow_pickle=True)['reconstruction']
    elif selecttwo==True:
        rec_1 = np.load(os.path.expanduser(path_1),allow_pickle=True)['reconstruction']
        rec_2 = np.load(os.path.expanduser(path_2),allow_pickle=True)['reconstruction']
        
    else:
        rec_1 = np.load(os.path.expanduser(path_1),allow_pickle=True)['reconstruction']
        rec_2 = np.load(os.path.expanduser(path_2),allow_pickle=True)['reconstruction']
        rec_3 = np.load(os.path.expanduser(path_3),allow_pickle=True)['reconstruction']
        if len(rec_1) != hp.nside2npix(NSIDE):
            rec_1 = hp.pixelfunc.ud_grade(rec_1,NSIDE)
        if len(rec_2) != hp.nside2npix(NSIDE):
            rec_2 = hp.pixelfunc.ud_grade(rec_2,NSIDE)
        if len(rec_3) != hp.nside2npix(NSIDE):
            rec_3 = hp.pixelfunc.ud_grade(rec_3,NSIDE)
    
    if selecttwo == False:
        recs = np.array([rec_1,rec_2,rec_3])
    if selecttwo == True:
        recs = np.array([rec_1,rec_2])
    
    frequency = freq*1e6
    ##location of fossil
    direction = np.radians([[110,70],[130,60],[110,40]])
    size_ang = np.radians([2,2,2])
    intensity_base = np.array([75,150,300])
    intensities_multiplier = (frequency/10e6)**(-0.5)
    intensities_fossil = intensity_base * intensities_multiplier
    model_I, model_x = funcs.make_model(NSIDE, frequency, object_intensity=intensities_fossil,object_size=size_ang,object_direction=direction)
    
    
    if selecttwo==False:
        noise_1 = abs(abs(rec_1)-abs(model_I))
        noise_2 = abs(abs(rec_2)-abs(model_I))
        noise_3 = abs(abs(rec_3)-abs(model_I))
    if selecttwo==True:
        noise_1 = abs(abs(rec_1)-abs(model_I))
        noise_2 = abs(abs(rec_2)-abs(model_I))
    
    if type(mask) != type(None):
        masked_pix = hp.query_strip(hp.npix2nside(len(noise_1)),np.radians(mask[0]),np.radians(mask[1]))
        if selecttwo==False:
            noise_1[masked_pix] = 0 
            noise_2[masked_pix] = 0 
            noise_3[masked_pix] = 0 
        if selecttwo==True:
            noise_1[masked_pix] = 0 
            noise_2[masked_pix] = 0 
            noise = np.array([noise_1,noise_2])
    
    if selecttwo==False:
            noise = np.array([noise_1,noise_2,noise_3])
    if selecttwo==True:
            noise = np.array([noise_1,noise_2])

    
    if type(degradeNSIDE) != type(None):
        if selecttwo==False:
            degrade_1 = hp.ud_grade(noise_1,degradeNSIDE)
            degrade_2 = hp.ud_grade(noise_2,degradeNSIDE)
            degrade_3 = hp.ud_grade(noise_3,degradeNSIDE)
            degraded = np.array([degrade_1,degrade_2,degrade_3])
            means = np.array([np.mean(degrade_1), np.mean(degrade_2), np.mean(degrade_3)])
            return noise, recs, degraded,means, model_I

        if selecttwo==True:
            degrade_1 = hp.ud_grade(noise_1,degradeNSIDE)
            degrade_2 = hp.ud_grade(noise_2,degradeNSIDE)
            degraded = np.array([degrade_1,degrade_2])
            means = np.array([np.mean(degrade_1), np.mean(degrade_2)])
            return noise, recs, degraded,means, model_I
    else:
        return noise, recs, model_I

#function to make the angular spectra 
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
        power[i] = spectrum[i] * i*(i + 1)
    return power


freqs = [10,15,20,25,30]
deg_N = 2**5
spectrasize = 400
spectrasize_high = 800
mask = np.array([0,91])

degraded_maps_100 = np.zeros([len(freqs),hp.pixelfunc.nside2npix(deg_N)])
degraded_maps_500 = np.zeros([len(freqs),hp.pixelfunc.nside2npix(deg_N)])
degraded_maps_1000 = np.zeros([len(freqs),hp.pixelfunc.nside2npix(deg_N)])
mean_100 = np.zeros(len(freqs))
mean_500 = np.zeros(len(freqs))
mean_1000 = np.zeros(len(freqs))
mean_noise = np.zeros([len(freqs),3])
spectral = np.zeros([len(freqs),3,spectrasize])
noisemap_undegraded = np.zeros([len(freqs),2,hp.pixelfunc.nside2npix(2**8)])
spectral_high = np.zeros([len(freqs),2,spectrasize_high])

for i in tqdm(range(len(freqs))):
    noise,recs,degraded,means,model_I = noise_map(freqs[i],degradeNSIDE=deg_N,mask=mask)
    degraded_maps_100[i] = degraded[0] 
    degraded_maps_500[i] = degraded[1]
    degraded_maps_1000[i] = degraded[2]
    mean_noise[i] = means
    noisemap_undegraded[i] = noise_map(freqs[i],selecttwo=True,mask=mask)[0]
    
    mean_100[i] = means[0]
    mean_500[i] = means[1]
    mean_1000[i] = means[2]
    
    
    for j in range(len(noise)):
        spectral[i][j] = spectra(noise[j],mask=mask,size=spectrasize)
    for k in range(len(spectral_high[1])):
        spectral_high[i][k] = spectra(noisemap_undegraded[i][k],mask=mask,size=spectrasize_high)
    
    
    
#full sky noise
noise,recs,degraded,means,model_I = noise_map(freqs[2],degradeNSIDE=deg_N,mask=None)
hp.mollview(np.log10(abs(degraded[1])),title='Residuals at {}MHz and 500.000 Baselines (Log scale)'.format(freqs[2]))



#full_sky_noise
noisefactor_5 = degraded_maps_500/degraded_maps_100
noisefactor_10 = degraded_maps_1000/degraded_maps_100
noisefactor_5_10 = degraded_maps_1000/degraded_maps_500
noise_map_100 = np.sort(degraded_maps_100)
noise_map_500 = np.sort(degraded_maps_500)
noise_map_1000 = np.sort(degraded_maps_1000)


#masked map


NPIX_masked = len(degraded[1])
NSIDE_masked = hp.pixelfunc.npix2nside(NPIX_masked)
masked = hp.query_strip(NSIDE_masked,np.radians(0),np.radians(91))
model_I_masked = np.copy(degraded[1])
model_I_masked[masked] = -1.6375e+30

fig, (ax1,ax2) = plt.subplots(ncols=2,tight_layout=True,figsize=(10,5))
plt.axes(ax1)
hp.mollview(np.log10((model_I_masked)),hold=True,title='masked map',cbar=False)

plt.axes(ax2)
plt.plot(noise_map_500[0])
#plt.axvline(x=6500,color='r')
plt.axvline(x=12000,color='r')
plt.title('selection of (averaged) pixels')
plt.ylabel('Average noise')
plt.show()


#histograam
mean_noise = np.zeros([len(noise_map_100),3])
error = np.zeros(np.shape(mean_noise))
selection = [2000,12000]
for j in range(len(mean_noise)):
    tot1 = (noise_map_100[j][selection[0]:selection[1]])
    tot2 = (noise_map_500[j][selection[0]:selection[1]])
    tot3 = (noise_map_1000[j][selection[0]:selection[1]])
    tot1[tot1==0] = np.NaN
    tot2[tot2==0] = np.NaN
    tot3[tot3==0] = np.NaN
    error[j][0] = np.nanstd(tot1)
    error[j][1] = np.nanstd(tot2)
    error[j][2] = np.nanstd(tot3)
    mean_noise[j][0] = np.nanmean(tot1)
    mean_noise[j][1] = np.nanmean(tot2)
    mean_noise[j][2] = np.nanmean(tot3)
    

    
    
mean_noise[mean_noise==0] =np.nan
width = 1/5-(0.1/5)
labels = ['10MHz','15MHz','20MHz','25MHz','30MHz']
x=np.arange(5)
x_error = np.ndarray.flatten(np.array([x-width*2,x,x+width*2]))
y_error = np.ndarray.flatten(np.transpose(mean_noise))
errors = np.ndarray.flatten(error)

fig,ax = plt.subplots()
rects1 = ax.bar(x-width*2,mean_noise[:,0],width,label='100.000')
rects2 = ax.bar(x-width,mean_noise[:,0]*(1/np.sqrt(5)),width,label='100.000 / \u221A5')
rects3 = ax.bar(x,mean_noise[:,1],width,label='500.000')
rects4 = ax.bar(x+width,mean_noise[:,0]*(1/np.sqrt(10)),width,label='100.000 / \u221A10')
rects5 = ax.bar(x+width*2,mean_noise[:,2],width,label='1.000.000')

ax.errorbar(x_error,y_error,yerr=errors,fmt='o',color='r')

ax.set_ylabel('Average noise in Jansky')
ax.set_title('Noise comparison')
ax.set_xticks(x, labels)
ax.legend()

fig.tight_layout()

plt.show()




    
##spectra

spectral_select = spectral[0][:,:]
spectral_angle = 180/np.arange(len(spectral_select[0]))
spectral_angle[spectral_angle == np.Inf] = 360
spectral_angle_high = 180/np.arange(spectrasize_high)
spectral_angle_high[spectral_angle_high == np.Inf] = 360


plt.plot(spectral_angle[200:],np.log10(spectral_select[0][200:]),label='100.000')
plt.plot(spectral_angle[200:],np.log10(spectral_select[1][200:]),label='500.000')
plt.plot(spectral_angle[200:],np.log10(spectral_select[2][200:]),label='1.000.000')
plt.gca().invert_xaxis()

plt.legend()
plt.show()




##spectrum of CL
look_at = [0,len(spectral_angle_high)-1]

test = np.mean(spectral_high[1][1]/spectral_high[1][0])
#test_2 = 1/np.sqrt(2000) * test
test_2 = 1/2000 * test


fig,ax = plt.subplots(constrained_layout=True)
ax.plot(np.log10(spectral_high[1][0]),label='100.000')
#ax.plot(np.log10(spectral_high[1][0]*(1/np.sqrt(5))),label='100.000 / \u221A5')
ax.plot(np.log10(spectral_high[1][0]*(1/5)),label='100.000 / 5')

ax.plot(np.log10(spectral_high[1][1]),label='500.000')
#ax.plot(np.log10(spectral_high[1][0]*(1/np.sqrt(10000))),label='100.000 / \u221A10.000')
ax.plot(np.log10(spectral_high[1][0]*(1/10000)),label='100.000 / 10.000')

#ax.plot(np.log10(spectral_high[1][0]*test_2),label='500.000 / 100.000 / \u221A2000')
ax.plot(np.log10(spectral_high[1][0]*test_2),label='500.000 / 100.000 / 2000')

ax.set_xlabel('order of l')
ax.set_ylabel('log10 Of Cl')
plt.title('angular spectral density of the noise at {}MHz'.format(freqs[1]))

plt.legend()
plt.show()

power_100 = spectopower(spectral_high[1][0])
power_500 = spectopower(spectral_high[1][1])
#power_predicted = power_500 * (test/np.sqrt(2000))
power_predicted = power_500 * (test/2000)


fig,ax = plt.subplots(constrained_layout=True)
ax.plot(np.log10(power_100),label='100.000')
#ax.plot(np.log10(power_100*(1/np.sqrt(5))),label='100.000 / \u221A5')
ax.plot(np.log10(power_100*(1/5)),label='100.000 / 5')
ax.plot(np.log10(power_500),label='500.000')
#ax.plot(np.log10(power_100*(1/np.sqrt(10000))),label='predicted after 4 years (theoretical)')
ax.plot(np.log10(power_100*(1/10000)),label='predicted after 4 years (theoretical)')

ax.plot(np.log10(power_100*test_2),label='predicted after 4 years (calculated through)')
ax.set_xlabel('order of $\ell$')
ax.set_ylabel('$log_{10} \ell(\ell+1)$Cl')
plt.title('angular spectral density of the noise at {}MHz'.format(freqs[1]))
plt.legend()
plt.show()

fig, ax = plt.subplots(nrows=2,ncols=1,tight_layout=True)
ax[0].plot(np.log10(spectral_high[1][0]),label='_100.000')
ax[0].plot(np.log10(spectral_high[1][0]*(1/np.sqrt(5))),label='_100.000 / \u221A5')
ax[0].plot(np.log10(spectral_high[1][1]),label='_500.000')
ax[0].plot(np.log10(spectral_high[1][0]*(1/np.sqrt(10000))),label='_100.000 / \u221A10.000')
ax[0].plot(np.log10(spectral_high[1][0]*test_2),label='_500.000 / 100.000 / \u221A2000')
ax[0].set_xlabel('order of l')
ax[0].set_ylabel('$\log_{10}$ Of Cl')
ax[1].plot(np.log10(power_100),label='100.000')
ax[1].plot(np.log10(power_100*(1/np.sqrt(5))),label='100.000 / \u221A5')
ax[1].plot(np.log10(power_500),label='500.000')
ax[1].plot(np.log10(power_100*(1/np.sqrt(10000))),label='100.000 / \u221A10.000')
ax[1].plot(np.log10(power_100*test_2),label='500.000 / 100.000 / \u221A2000')
ax[1].set_xlabel('order of $\ell$')
ax[1].set_ylabel('$log_{10} \ell(\ell+1)$Cl')
plt.suptitle('angular spectral density of the noise at {}MHz'.format(freqs[1]))
fig.legend(bbox_to_anchor =(1.4, 0.7))
plt.show()



#big plot 
Jansky=True

spectral_high_p = np.zeros(np.shape(spectral_high))
if Jansky == True:
    for k in range(len(spectral_high)):
        for l in range(len(spectral_high[0][0])):
            spectral_high_p[k][0][l] = spectral_high[k][0][l] *(l*(l+1))
            spectral_high_p[k][1][l] = spectral_high[k][1][l] *(l*(l+1))


fig,((ax1,ax2),(ax3,ax4),(ax5,ax6))=plt.subplots(nrows=3,ncols=2,sharex=True,tight_layout=True,figsize=(7, 5))

ax1.plot(np.log10(spectral_high_p[0][0]),label='100.000')
#ax1.plot(np.log10(spectral_high_p[0][0]*(1/np.sqrt(5))),label='100.000 / \u221A5')
ax1.plot(np.log10(spectral_high_p[0][0]*(1/5)),label='100.000 / 5')
ax1.plot(np.log10(spectral_high_p[0][1]),label='500.000')
#secaxis=ax1.secondary_xaxis('top')
ax1.set_title('10MHz')
#ax1 = plt.gca()
#ax1.invert_xaxis()


ax2.plot(np.log10(spectral_high_p[1][0]),label='')
#ax2.plot(np.log10(spectral_high_p[1][0]*(1/np.sqrt(5))),label='')
ax2.plot(np.log10(spectral_high_p[1][0]*(1/5)),label='')
ax2.plot(np.log10(spectral_high_p[1][1]),label='')
#secaxis=ax2.secondary_xaxis('top')
#ax2 = plt.gca()
#ax2.invert_xaxis()
ax2.set_title('15MHz')

ax3.plot(np.log10(spectral_high_p[2][0]),label='')
#ax3.plot(np.log10(spectral_high_p[2][0]*(1/np.sqrt(5))),label='')
ax3.plot(np.log10(spectral_high_p[2][0]*(1/5)),label='')
ax3.plot(np.log10(spectral_high_p[2][1]),label='')
#secaxis=ax3.secondary_xaxis('top')
#ax3 = plt.gca()
#ax3.invert_xaxis()
ax3.set_title('20MHz')



ax4.plot(np.log10(spectral_high_p[3][0]),label='')
#ax4.plot(np.log10(spectral_high_p[3][0]*(1/np.sqrt(5))),label='')
ax4.plot(np.log10(spectral_high_p[3][0]*(1/5)),label='')
ax4.plot(np.log10(spectral_high_p[3][1]),label='')
#secaxis=ax4.secondary_xaxis('top')
#ax4 = plt.gca()
#ax4.invert_xaxis()
ax4.set_title('25MHz')



ax5.plot(np.log10(spectral_high_p[4][0]),label='')
#ax5.plot(np.log10(spectral_high_p[4][0]*(1/np.sqrt(5))),label='')
ax5.plot(np.log10(spectral_high_p[4][0]*(1/5)),label='')
ax5.plot(np.log10(spectral_high_p[4][1]),label='')
#secaxis=ax5.secondary_xaxis('top')
#ax5 = plt.gca()
#ax5.invert_xaxis()
ax5.set_title('30MHz')

fig.delaxes(ax6)
fig.legend(loc=4,bbox_to_anchor=(0.91,0.08))
fig.text(0.5, 0.01, '$\ell$', ha='center')
fig.text(0.00, 0.5, '$log_{10} \ell(\ell+1)C_\ell$', va='center', rotation='vertical')

fig.suptitle('angular spectral noise')
    
plt.show()


plt.plot(spectral_angle_high,np.log10(spectral_high[0][0]),label='100.000 at 10MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[0][1]),label='500.000 at 10MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[1][0]),label='100.000 at 15MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[1][1]),label='500.000 at 15MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[2][0]),label='100.000 at 20MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[2][1]),label='500.000 at 20MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[3][0]),label='100.000 at 25MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[3][1]),label='500.000 at 25MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[4][0]),label='100.000 at 30MHz')
plt.plot(spectral_angle_high,np.log10(spectral_high[4][1]),label='500.000 at 30MHz')
plt.gca().invert_xaxis()
plt.xlim(5,0)
plt.ylabel('$log_{10} C_\ell$ (in Jansky)')
plt.xlabel('Angle of $C_\ell$')
plt.legend(bbox_to_anchor =(1.02, 0.9))
plt.title('Angular power spectrum of the noise')
plt.show()



##spectra fake sources
NSIDE = 2**8
frequency = 15e6
spectra_length =800
models = np.zeros((4,hp.nside2npix(NSIDE)))
models_single = np.zeros((4,hp.nside2npix(NSIDE)))
models2 = np.zeros((4,hp.nside2npix(NSIDE)))
models_single2 = np.zeros((4,hp.nside2npix(NSIDE)))
model_x = funcs.make_model(NSIDE, frequency)[1]
sources = np.radians(np.array([[90,0],[140,110],[110,-70],[120,120]]))
x_hats = hp.ang2vec(sources[:,0], sources[:,1])
size_ang = np.radians(1)
spectra =np.zeros((4,spectra_length+1))
spectra2=np.zeros((4,spectra_length+1))
spectra_single = np.zeros((4,spectra_length+1))
spectra_single2 = np.zeros((4,spectra_length+1))
sources_power = [100,200]
model_point = np.zeros(hp.nside2npix(NSIDE))
model_point[int(len(model_point)*2/3)] = 30
spectra_point = hp.sphtfunc.anafast(model_point,lmax=spectra_length)
power_point = spectopower(spectra_point)

for i in range(len(models)):
    model_temp = funcs.gaussian_blob(sources_power[0],model_x, sources[i], size_ang)
    models_temp2 = funcs.gaussian_blob(sources_power[1],model_x, sources[i], size_ang)
    if i == 0:
        models[i] = model_temp
        models2[i] = models_temp2
    else:
        models[i] = models[i-1] + model_temp
        models2[i] = models2[i-1] + models_temp2
    models_single[i] = model_temp
    models_single2[i] =models_temp2
    
    spectra[i] = hp.sphtfunc.anafast(models[i],lmax=spectra_length)
    spectra_single[i] = hp.sphtfunc.anafast(models_single[i],lmax=spectra_length)
    spectra2[i] = hp.sphtfunc.anafast(models2[i],lmax=spectra_length)
    spectra_single2[i] = hp.sphtfunc.anafast(models_single2[i],lmax=spectra_length)

model_extended = funcs.gaussian_blob(sources_power[0],model_x, sources[0], np.radians(10))
spectrum_extended = hp.sphtfunc.anafast(model_extended,lmax=spectra_length)
power_extended = spectopower(spectrum_extended)

power_single = spectopower(spectra[0])
power_single_high = spectopower(spectra2[0])
power_double = spectopower(spectra[1])
power_different_loc = spectopower(spectra_single[2])

plt.plot(np.log10(power_single),label='single source at [0,0]')
#plt.plot(np.log10(power_single_high),label='two sources at [0,0] double power')
#plt.plot(np.log10(power_double),label='single source double')
plt.plot(np.log10(power_different_loc),label='single source at [-50,110]')
plt.plot(np.log10(spectrum_extended),label='extended source')
plt.plot(np.log10(power_predicted),label='500.000')
plt.plot(np.log10(power_point),label='point_source')
plt.xlabel('$\ell$')
plt.ylabel('$log_{10}\, \ell(\ell+1)\,C_\ell$')
plt.ylim(-9,2)
plt.legend(loc='lower right')
plt.show()

#milikelvin

mhz_15 = np.copy(spectral_high_p[1][1])
mhz_30 = np.copy(spectral_high_p[4][1])

mk_15 = np.sqrt((funcs.Tb(mhz_15,15,4*np.pi)/(2*np.pi))) * 1000 /(np.sqrt(2000))
mk_30 = np.sqrt((funcs.Tb(mhz_30,30,4*np.pi)/(2*np.pi))) * 1000 /(np.sqrt(2000))

plt.plot(np.log10(mk_15),label='15Mhz (z\u2248100)')
plt.plot(np.log10(mk_30),label='30MHz (z\u224850)')
plt.ylabel('$\log_{10}$ ($\ell(\ell+1)\,C_{\ell}$ / $2\pi)^{1/2}$ /mK')
plt.xlabel('$\ell$')
plt.title('Expected power spectra after 4 years')
plt.legend()
plt.show()

##system noise
freqs = np.linspace(0.5,50,100)*1e6
wavelengths = 299792458/freqs
skyrms = np.zeros(len(freqs))
temp = np.zeros(len(freqs))
for i in range(len(freqs)):
    skyrms[i] = funcs.rms_sky(wavelengths[i],bandwidth=100e3 , t=0.1)
    temp[i] = funcs.Tb(skyrms[i],freqs[i]/1e6,4*np.pi)

plt.plot(freqs/1e6,np.log10(skyrms),label='Jansky')
plt.plot(freqs/1e6,np.log10(temp),label='Kelvin')
plt.xlabel('Frequency in Mzh')
plt.ylabel('Log of noise')
plt.legend()
plt.show()



