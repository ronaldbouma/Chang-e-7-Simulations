#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:03:07 2023

@author: Ronald
"""
#script to simulate the effect of random position error on a point source.

#cd
import os
direct = os.getcwd()
if direct != '/Users/Ronald/Desktop/Thesis/Scripts':
    os.chdir('desktop/thesis/scripts')

#packages
import numpy as np
import matplotlib.pyplot as plt
import functions2 as funcs

def sphtocar(theta,phi):
    x = np.cos(phi)*np.sin(theta)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return np.array((x,y,z))

##constant
c = 299792458
r_moon = 1.7371e6
T_moon = 655.728*3600
w_moon = 2*np.pi/(T_moon)

#sat states
sat1_state = [2338,0.04,np.radians(76),np.pi,np.pi,0] #a,e,i,Omega,w,starting true anomaly
sat2_state = [2200,0.1,np.radians(88),np.pi,np.pi,0.3*np.pi]
ground_state = np.radians(170),np.radians(10)
ground_orien = np.radians([170,0])
sat1_orien = np.radians([0,0])
sat2_orien = np.radians([0,0])
mutation1 = np.radians([0.0000,0.0000])
mutation2 = np.radians([-0.0000,-0.0000])

freqs = [30e6,50e6]

bandwidth = 100e3
antenna_length = 5


#time selection
max_years = 2
t_int = 0.1
time_span = [0,max_years*31556926]

baseline_amount = 15000
target = (110,40)


t_points = funcs.select_visible(time_span, np.radians(target), int(baseline_amount), sat1_state, sat2_state, ground_state)
Ds = funcs.baselines(t_points, ground_state, sat1_state, sat2_state, ground_orien, sat1_orien, sat2_orien)
Positions = Ds[3:6]
orientations = Ds[6:]

#grid, functionally the same as grid reconstruct except it saves output for use later\
    #because the process takes quite a bit of time and i didnt want to repeat the process.

resolution = [80,80]
size = 2

theta = np.radians(np.linspace(90-size/2,90+size/2,resolution[0]))
phi = np.radians(np.linspace(90-size/2,90+size/2,resolution[1]))
    
for k in range(len(freqs)):
    frequency = freqs[k]
    print('starting {} MHz'.format(freqs[k]/1e6))
    wavelength = c/frequency
    model_x = np.zeros((resolution[0],resolution[0],3))
    for i in range(len(theta)):
        for j in range(len(phi)):
            model_x[i][j] = sphtocar(theta[i], phi[j])
            
    
    model_x_reshaped = np.reshape(model_x,(resolution[0]*resolution[1],3))
    model_I_grid = np.zeros((resolution))
    model_I_grid[int(resolution[0]/2),int(resolution[1]/2)]= 10
    #model_I_grid[2][12] = 100
    plt.imshow(model_I_grid)
    model_I = model_I_grid.flatten()
    eclipse = np.ones((baseline_amount,3), dtype=bool).transpose()
    
    
    Vs,baselines,Vs_nonoise,noise = funcs.Visibility_sampling(Positions,eclipse,model_x_reshaped,model_I,wavelength,Noise=[0,400])
    
    errors = np.linspace(0,5,100)
    maps = np.zeros((len(errors),resolution[0],resolution[1]))
    lines = np.zeros((len(errors),resolution[0]*resolution[1]))
    error_mean = np.zeros(len(errors))
    
    for i in range(len(errors)):
        print('')
        print('performing calculation {} of {}'.format(i+1,len(errors)))
        print('')
        error = errors[i]*(1/np.sqrt(3))*(np.random.randn(len(Positions),len(Positions[0]),3))
        Positions_werror = Positions + error
        error_mean[i] = np.mean(np.linalg.norm(error,axis=2))
    
        rec = funcs.reconstruct(Positions_werror, Vs, resolution[0]*resolution[1], model_x_reshaped, wavelength,divide_by_baselines=True)
        lines[i] = rec
        recon = np.reshape(rec,(resolution[0],resolution[1]))
        maps[i] = recon
        
    maximum = np.argmax(lines,axis=1)
    cut_off = np.sort(np.where(maximum != maximum[0]))
    critical_error = error_mean[cut_off]
    values = np.zeros(len(lines))
    point_source = np.argmax(lines[0])
    for j in range(len(values)):
        values[j] = lines[j][point_source] / lines[0][maximum[0]]
    
    mean_noise = np.mean(abs(lines),axis=1)
    mean_noise_rel = mean_noise/mean_noise[0]
    fit = np.polyfit(error_mean,values,3)
    yfit = np.polyval(fit, error_mean)
    wavelengths =  error_mean/wavelength
    
    
    filename = os.path.join("outputs/position_check_{}MHz.npz".format(int(frequency/1e6)))
    np.savez(filename, maps,maps,values=values,yfit=yfit,critical_error=critical_error,error_mean=error_mean,wavelengths=wavelengths)



# plot 1
# plt.plot(error_mean,values,label='relative strength point source to no error')
# plt.plot(error_mean,yfit,label='best fit')
# plt.xlabel('Mean Position error in meters')
# plt.ylabel('')
# plt.title('Position error effects for 15 MHz')
# plt.legend(loc='lower left')
# plt.show()


#save

filename = os.path.join("output/position_check_{}MHz".format(int(frequency/1e6)))
np.savez(filename, maps,maps,values=values,yfit=yfit,critical_error=critical_error,error_mean=error_mean,wavelengths=wavelengths)



#loading and inspecting the results.
    
path_10 = ('~/desktop/thesis/scripts/outputs/position_check_10MHz.npz')
path_15 = ('~/desktop/thesis/scripts/outputs/position_check_15MHz.npz')
path_20 = ('~/desktop/thesis/scripts/outputs/position_check_20MHz.npz')
path_25 = ('~/desktop/thesis/scripts/outputs/position_check_25MHz.npz')
path_30 = ('~/desktop/thesis/scripts/outputs/position_check_30MHz.npz')
path_50 = ('~/desktop/thesis/scripts/outputs/position_check_50MHz.npz')


data_10 = np.load(os.path.expanduser(path_10),allow_pickle=True)
data_15 = np.load(os.path.expanduser(path_15),allow_pickle=True)
data_20 = np.load(os.path.expanduser(path_20),allow_pickle=True)
data_25 = np.load(os.path.expanduser(path_25),allow_pickle=True)
data_30 = np.load(os.path.expanduser(path_30),allow_pickle=True)
data_50 = np.load(os.path.expanduser(path_50),allow_pickle=True)

values_10 = data_10['values']
values_15 = data_15['values']
values_20 = data_20['values']
values_25 = data_25['values']
values_30 = data_30['values']
values_50 = data_50['values']

fit_10 = data_10['yfit']
fit_15 = data_15['yfit']
fit_20 = data_20['yfit']
fit_25 = data_25['yfit']
fit_30 = data_30['yfit']
fit_50 = data_50['yfit']

error_10 = data_10['error_mean']
error_15 = data_15['error_mean']
error_20 = data_20['error_mean']
error_25 = data_25['error_mean']
error_30 = data_30['error_mean']
error_50 = data_50['error_mean']

crit_10 = data_10['critical_error']
crit_15 = data_15['critical_error']
crit_20 = data_20['critical_error']
crit_25 = data_25['critical_error']
crit_30 = data_30['critical_error']
crit_50 = data_50['critical_error']

crit_errors = np.array((crit_10[0][0],crit_15[0][0],crit_20[0][0],crit_25[0][0],crit_30[0][0],crit_50[0][0]))



#combined plot, can turn on and wether you want the fit, the raw results or both.

wavelengths = lambda lengths: lengths/wavelength
lengths =  lambda wavelengths: wavelength * wavelengths

#plt.plot(error_10,values_10,label='_10 MHz, simulation',color='b')
plt.plot(error_10,fit_10,label='10 MHz',color='b')
#plt.plot(error_15,values_15,label='_15 MHz, simulation',color='y')
plt.plot(error_15,fit_15,label='15 MHz',color='y')
#plt.plot(error_20,values_20,label='_20 MHz, simulation',color='g')
plt.plot(error_20,fit_20,label='20 MHz',color='g')
#plt.plot(error_25,values_25,label='_25 MHz, simulation',color='r')
plt.plot(error_25,fit_25,label='25 MHz',color='r')
#plt.plot(error_30,values_30,label='_30 MHz, simulation',color='c')
plt.plot(error_30,fit_30,label='30 MHz',color='c')
#plt.plot(error_50,values_50,label='50 MHz, simulation')
#plt.plot(error_50,fit_50,label='50 MHz')
#plt.vlines(crit_errors,0,1)
plt.xlabel('Mean error in meters')
plt.ylabel('relative visibility')
#plt.xlim([0,1.5])
#plt.ylim([0.5,1.1])
plt.legend()
plt.show()



#second plot, detailed look at single frequency
wavelengths = data_15['wavelengths']
critical_error = data_15['critical_error']
wavelength =  c/15e6

wavelengths = lambda lengths: lengths/wavelength
lengths =  lambda wavelengths: wavelength * wavelengths

fig, ax = plt.subplots(constrained_layout=True)
ax2 = ax.secondary_xaxis("top", functions=(wavelengths, lengths))
ax.plot(error_15,values_15,label='simulation')
ax.plot(error_15,fit_15,label='best fit')
ax.axvline(critical_error[0][0],color='r',label='critical error')
ax.set_xlabel('Mean Position error in meters')
ax2.set_xlabel('Wavelengths')
ax.set_ylabel('relative strength of point source')
plt.title('Position error effects for 15 MHz')
plt.legend(loc='lower left')
plt.show()















