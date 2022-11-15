#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:44:09 2021

@author: Ronald
"""

import numpy as np
from scipy.spatial.transform import Rotation
import time
from tqdm import tqdm
import healpy as hp
import numexpr as ne
import pygdsm


def Sv(Tb_1e6,v_mhz,Omega): #function to convert kelvin to intensity
    return 0.93e3 * Tb_1e6 * Omega * (v_mhz/10)**2

def Tb(Sv,v_mhz,Omega): #same as above but reverse
    return Sv * 0.93e3**(-1) * Omega**(-1)*(v_mhz/10)**(-2)

def T_sky(v_mhz): #sky temperature 
    if v_mhz > 2:
        sky = 16.3*(v_mhz/2)**(-2.53)
    elif v_mhz <= 2:
        sky = 16.3*(v_mhz/2)**(-0.3)
    return sky

def rms_sky(wavelength ,bandwidth=10e3 , t=1, Sr=2*np.pi,  A_length=5):
    #function to generate the RMS sky noise value at a given wavelength in jansky
    #t value can be set to change integration time
    #A_length is the length of the antenna
    if wavelength > A_length:
        An = (2*A_length)**2 / 4
    else:
        An = wavelength**2 / 4
    frequency = 299792458/wavelength
    Tsky = T_sky(frequency/1e6)
    return np.sqrt(2)*(Tsky*1e6)/(An*np.sqrt(t*bandwidth)) * 1.38e-23 * 1e26


#sat state evolve
def state(x,u=4.9048695e3):
    #takes keplerian elements and returns state vector, standard gravity parameter takes km^3 as value
    #input are as [semi-major axis,eccentricity,inclination,right ascension of the ascending node, argument of perigee, True anomaly]
    mu = u #gravitational parameter
    a,e,i,Omega,w,theta = x[0],x[1],x[2],x[3],x[4],x[5] #extracting elements
    r = (a*(1-e**2)) / (1 + e*np.cos(theta))
    nu = np.array([r*np.cos(theta),r*np.sin(theta)])
    l1 = np.cos(Omega)*np.cos(w)-np.sin(Omega)*np.sin(w)*np.cos(i)
    l2 = -1*np.cos(Omega)*np.sin(w)-np.sin(Omega)*np.cos(w)*np.cos(i)
    m1 = np.sin(Omega)*np.cos(w)+np.cos(Omega)*np.sin(w)*np.cos(i)
    m2 = -1*np.sin(Omega)*np.sin(w)+np.cos(Omega)*np.cos(w)*np.cos(i)
    n1 = np.sin(w)*np.sin(i)
    n2 = np.cos(w)*np.sin(i)
    B = np.array([l1,l2,m1,m2,n1,n2]).reshape(3,2) #creating matrix
    pos = np.dot(B,nu) #positions
    H = np.sqrt(mu*a*(1-e**2))
    xdot = mu/H * (-1*l1 * np.sin(theta) + l2*(e+np.cos(theta))) 
    ydot = mu/H * (-1*m1*np.sin(theta) + m2*(e+np.cos(theta)))
    zdot = mu/H * (-1*n1*np.sin(theta) + n2*(e+np.cos(theta)))
    state = np.append(pos,[xdot,ydot,zdot])
    return state

def getanomaly(M,e,init=3.14): #takes the mean anomaly and returns the true anomaly
    from scipy.optimize import fsolve
    def func(E,M,e):
        return E-e*np.sin(E)-M
    E = fsolve(func,init,args=(M,e))
    theta = 2*(np.arctan((np.sqrt((1+e)/(1-e)) * np.tan(E))))[0]
    return theta

def mean_anomaly(t,a,tau=0,u=4.9048695e3):
    #for use in state_evolve
    n = np.sqrt(u/(a**3))
    return n*(t-tau)

def state_evolve(t_range,ground_state,sat_state1,sat_state2):
    #simple state evolution of a sattelite using only keplerian dynamics
    Mean_anomalies1 = mean_anomaly(t_range,sat_state1[0])
    Mean_anomalies2 = mean_anomaly(t_range,sat_state2[0])
    True_anomalies1, True_anomalies2 = np.array([]), np.array([])
    for i in range(len(t_range)): #for some reason it doesnt like to take the array in one go
        True_anomalies1 = np.append(True_anomalies1,getanomaly(Mean_anomalies1[i], sat_state1[1]))
        True_anomalies2 = np.append(True_anomalies2,getanomaly(Mean_anomalies2[i], sat_state2[1]))
    
    sat_pos = np.array([])
    sat_pos2 = np.array([])
    ground_pos = np.array([])
    
    for j in range(len(t_range)):
        sat_elements = np.append(sat_state1[:5],True_anomalies1[j])
        sat_elements2 = np.append(sat_state2[:5],True_anomalies2[j])
        spos = state(sat_elements)[:3]
        spos2 = state(sat_elements2)[:3]
        gpos = lunar_rot(ground_state[0],ground_state[1],t_range[j])
        sat_pos = np.append(sat_pos,spos)
        sat_pos2 = np.append(sat_pos2,spos2)
        ground_pos = np.append(ground_pos,gpos)
    ground_pos=np.reshape(ground_pos,(int(len(sat_pos)/3),3)) #reshape 
    sat_pos=np.reshape(sat_pos,(int(len(sat_pos)/3),3))*1000 #reshape and change to meters
    sat_pos2=np.reshape(sat_pos2,(int(len(sat_pos2)/3),3))*1000
    return ground_pos, sat_pos, sat_pos2, t_range


def eclipse_check_hat(A,source_hat):
    #function to check if a body is eclipsing a certain direction
    #mostly for use with the larger eclipse checker functions
    r_sat = np.linalg.norm(A)
    r_hat_sat = np.array([A[0],A[1],A[2]])/r_sat
    cos_psi = np.matmul(source_hat,r_hat_sat)
    a = r_sat * np.sin(np.arccos(cos_psi))
    return cos_psi, a

def eclipse_checker(Positions,model_x,R_m=1.7371e6,full_output=False,bars=False):
    #takes all positions from the central body to check what directions are eclipsed
    #inputs are the satellite positions (x,y,z), assumes a array of shape (3,number of positions, 3)
    #model_x are the directions you want to check (x_hat,y_hat,z_hat)
    #outputs a true false array for each sattelite position True means visible false means not visible 
    #output array is shape is (3,number of positions, number of directions)
    #activate timer
    t_0 = time.time()
    #generating array to be filled
    Full_array = np.zeros([3,len(Positions[0]),len(model_x)],dtype=bool)
    #print messages
    print('making eclipse array')
    print()
    #generates the eclipse array
    for i in tqdm(range(len(Positions[0])),disable=bars):
        g = eclipse_check_hat(Positions[0][i], model_x)
        s1 = eclipse_check_hat(Positions[1][i], model_x)
        s2 = eclipse_check_hat(Positions[2][i], model_x)
        ground_vis = ~((g[0]<=0) * (g[1]<=R_m))
        s1_vis = ~((s1[0]<=0) * (s1[1]<=R_m))
        s2_vis = ~((s2[0]<=0) * (s2[1]<=R_m))
        Full_array[0][i] = ground_vis * s1_vis
        Full_array[1][i] = ground_vis * s2_vis
        Full_array[2][i] = s1_vis * s2_vis
    
    #if full_output=True also generates a mask to be used for healpix
    if full_output==True:
        eclipser = np.zeros(len(model_x))
        print()
        print('generating masks')
        for k in tqdm(range(len(Full_array[0])),disable=True):
            eclipser += 1*Full_array[0,k,:] + 1*Full_array[1,k,:] + 1*Full_array[2,k,:]
        participating_x = (eclipser != 0)
        always_visible_x = (eclipser == 3*len(Full_array[0]))
        t_end=time.time()
        print('eclipse check operation took',round(t_end-t_0,1),'seconds')
        return Full_array, eclipser, participating_x, always_visible_x
    else:
        t_end=time.time()
        print('eclipse check operation took',round(t_end-t_0,1),'seconds')
        return Full_array


def eclipse_checker_single(Positions,model_x,R_m=1.7371e6):
    #takes a single 3 set of positions and gives the eclipsing array
    #for use in visibility sampling function
    #the full eclipse_checker can use too much memory, this is less efficient time wise but less memory intensive
    #inputs are the satellite position (x,y,z), assumes a array of shape (x,y,z)
    #and the directions you want to check (x_hat,y_hat,z_hat)
    #outputs a true false array for each sattelite position True means visible false means not visible 
    #output array is shape is (3, number of directions)
    #generating array to be filled
    Full_array_single = np.zeros([3,len(model_x)],dtype=bool)
    #generates the eclipse array
    g = eclipse_check_hat(Positions[0], model_x)
    s1 = eclipse_check_hat(Positions[1], model_x)
    s2 = eclipse_check_hat(Positions[2], model_x)
    ground_vis = ~((g[0]<=0) * (g[1]<=R_m))
    s1_vis = ~((s1[0]<=0) * (s1[1]<=R_m))
    s2_vis = ~((s2[0]<=0) * (s2[1]<=R_m))
    Full_array_single[0] = ground_vis * s1_vis
    Full_array_single[1] = ground_vis * s2_vis
    Full_array_single[2] = s1_vis * s2_vis
    return Full_array_single


def select_visible(t_range, source_loc, amount, sat1_state, sat2_state,ground_state):
    #selects a certain number of baseline that can all "see" the source 
    #t_range = time range, source_loc = location of souce in radians, amount=number of baselines you want
    #sat states are the initial positions of the satelites (Keplerian elements, see state_evolve)
    print('selecting baselines')
    print('')
    R_m = 1.7371e6
    source_xyz = hp.ang2vec(source_loc[0], source_loc[1])
    count = 0
    t_useful = np.array([])
    loop=0
    while count < amount:
        t_points = np.random.randint(t_range[0],t_range[1],int(amount/10))
        Positions = baselines(t_points, ground_state, sat1_state, sat2_state, [0,0], [0,0], [0,0])[3:6]
        for i in range(len(Positions[0])):
            g = eclipse_check_hat(Positions[0][i], source_xyz)
            s1 = eclipse_check_hat(Positions[1][i], source_xyz)
            s2 = eclipse_check_hat(Positions[2][i], source_xyz)
            ground_vis = ~((g[0]<=0) * (g[1]<=R_m))
            s1_vis = ~((s1[0]<=0) * (s1[1]<=R_m))
            s2_vis = ~((s2[0]<=0) * (s2[1]<=R_m))
            if ground_vis*s1_vis*s2_vis == True:
                t_useful = np.append(t_useful,t_points[i])
        count = len(np.unique(t_useful))
        loop += 1
        if loop > 100: #set a max number of tries here
            print('source not visible at enough points')
            break
    return (np.unique(t_useful)[:amount]).astype(int)


def baselines(t_range,ground_state,sat1_state,sat2_state,ground_orien_init,sat1_orien,sat2_orien,sat1_mut=[0,0],sat2_mut=[0,0],full_output=False):
    #function te generate all the baselines given the sattelite and ground initial position and at points t_range
    #get baselines is also inculded the visibility sampling function, so usually this function is not needed on its own
    ground_pos,sat_pos1,sat_pos2,t_points = state_evolve(t_range,ground_state,sat1_state,sat2_state)
    
    #initializing arrays
    B_g_s1 = np.array([])
    B_g_s2 = np.array([])
    B_s1_s2 = np.array([])
    
    ground_orien = np.array([])
    sat1_orientation = np.array([])
    sat2_orientation = np.array([])
    
    #evolve the system
    for j in range(len(t_points)):
        ground = (ground_orien_init[0] ,(t_points[j] * 2*np.pi/(655.728*3600) + ground_orien_init[1]))
        ground_xyz = np.cos(ground[1])*np.sin(ground[0]),np.sin(ground[0])*np.sin(ground[1]), np.cos(ground[0])
        ground_orien = np.append(ground_orien,ground_xyz)
        s1 = (sat1_orien[0]+t_points[j]*sat1_mut[0]), (sat1_orien[1]+t_points[j]*sat1_mut[1])
        s1_xyz = np.cos(s1[1])*np.sin(s1[0]),np.sin(s1[0])*np.sin(s1[1]), np.cos(s1[0])
        sat1_orientation = np.append(sat1_orientation,s1_xyz)
        s2 = (sat2_orien[0]+t_points[j]*sat2_mut[0]), (sat2_orien[1]+t_points[j]*sat2_mut[1])
        s2_xyz = np.cos(s2[1])*np.sin(s2[0]),np.sin(s2[0])*np.sin(s2[1]), np.cos(s2[0])
        sat2_orientation = np.append(sat2_orientation,s2_xyz)
    

    ground_orien = np.reshape(ground_orien,(int(len(ground_orien)/3),3))
    sat1_orientation = np.reshape(sat1_orientation,(int(len(sat1_orientation)/3),3))
    sat2_orientation = np.reshape(sat2_orientation,(int(len(sat2_orientation)/3),3))
    
    for i in range(len(sat_pos1)):
        B_g_s1 = np.append(B_g_s1,(ground_pos[i]-sat_pos1[i]))
        B_g_s2 = np.append(B_g_s2,(ground_pos[i]-sat_pos2[i]))
        B_s1_s2 = np.append(B_s1_s2,(sat_pos1[i]-sat_pos2[i]))
    
    if full_output==True:
        B_g_s1 = np.append(B_g_s1,-B_g_s1)
        B_g_s2 = np.append(B_g_s2,-B_g_s2)
        B_s1_s2 = np.append(B_s1_s2,-B_s1_s2)
    
    B_g_s1 = np.reshape(B_g_s1,(int(len(B_g_s1)/3),3))
    B_g_s2 = np.reshape(B_g_s2,(int(len(B_g_s2)/3),3))
    B_s1_s2 = np.reshape(B_s1_s2,(int(len(B_s1_s2)/3),3))
    
    return np.asarray([B_g_s1,B_g_s2,B_s1_s2,ground_pos,sat_pos1,sat_pos2,ground_orien,sat1_orientation,sat2_orientation])

#transforms
def galactic_transform(h_a,dec):
    #convert galactic coordinates to standard angles 
    phi = np.radians(h_a[0]*15 + h_a[1]/60 + h_a[2]/3600)
    if dec[0] >= 0:
        theta = abs(np.radians(dec[0] + dec[1]/60 + dec[2]/3600 -90))
    elif dec[0] < 0:
        theta = np.radians(abs(dec[0]) + dec[1]/60 + dec[2]/3600 +90)
    return [theta,phi]

def rot_transform(orien,target):
    #transforms spherical coordinates from one set to another
    a = np.sin(orien[0])*np.cos(orien[1]),np.sin(orien[0])*np.sin(orien[1]),np.cos(orien[0])
    target_xyz = np.sin(target[0])*np.cos(target[1]),np.sin(target[0])*np.sin(target[1]),np.cos(target[0])
    k = np.cross([0,0,1], a)
    if np.linalg.norm(k)==0:
        new_target=target
    else:
        khat = k/np.linalg.norm(k)
        beta = np.arccos(np.dot([0,0,1],a))
        r = np.linalg.inv(Rotation.from_rotvec(beta*khat).as_matrix())
        #r = Rotation.from_rotvec(beta*khat).as_matrix()
        new_target_xyz = np.dot(r,target_xyz)
        if abs(new_target_xyz[1]) < 1e-13:
            new_target_xyz[1] = 0
            new_target = np.arccos(new_target_xyz[2]/(np.linalg.norm(new_target_xyz))), \
                np.arctan2(new_target_xyz[1],new_target_xyz[0])
        else:
            new_target = np.arccos(new_target_xyz[2]/(np.linalg.norm(new_target_xyz))), \
                np.arctan2(new_target_xyz[1],new_target_xyz[0])
    return new_target
    

    
def dipole_inter(target_theta1,target_theta2,A0=1):
    if target_theta1 == 0:
        A1=0
    else:
        A1 = (np.cos(np.pi/2 * np.cos(target_theta1))/np.sin(target_theta1))**2
    if target_theta2 == 0:
        A2 = 0
    else:
        A2 = (np.cos(np.pi/2 * np.cos(target_theta2))/np.sin(target_theta2))**2
    return np.sqrt(A1*A2) * A0

def dipole_An(orientation1,orientation2,model_x):
    A1 = abs(1-np.dot(model_x,orientation1))
    A2 = abs(1-np.dot(model_x,orientation2))
    return A1*A2

def lunar_rot(theta,phi,t,w_moon=2*np.pi/(655.728*3600),r_moon=1.7371e6):
    #gives lunar based coordinates after initial time
    x=r_moon*np.sin(theta)*np.cos(phi+w_moon*t)
    y=r_moon*np.sin(theta)*np.sin(phi+w_moon*t)
    z=r_moon*np.cos(theta)
    return np.array([x,y,z])

def sph_to_cart(A):
    #transforms spherical to carthesian coordinates (normalized)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for i in range(len(A)):
        theta,phi = A[i][0],A[i][1]
        x,y,z = np.append(x,np.cos(phi)*np.sin(theta)),\
            np.append(y,np.sin(theta)*np.sin(phi)), np.append(z,np.cos(theta))
    xyz = np.column_stack((np.column_stack((x,y)),z))

    return xyz

def cart_to_sph(X_hats,lat=False):
    x = X_hats[:,0]
    y = X_hats[:,1]
    z = X_hats[:,2]
    theta = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
    phi = np.arctan2(y, x)
    if lat==True:
        for i in range(len(theta)):
            if theta[i]<=(np.pi/2):
                theta[i]=abs(theta[i]-(np.pi/2))
            elif theta[i]>=(np.pi/2):
                theta[i] = -1*(theta[i]-(np.pi/2))
        theta = np.degrees(theta)
        phi = np.degrees(phi)
    
    return np.transpose((theta,phi))


def uvw_transform(phase_target,Positions,wavelength, grid_resolution=500, psf_output = False):
    #takes a direction on the sky and a set of satellite positions and returns the uvw values for those positions
    #outputs a single array with uvw values as well as arrays containing all combined u, v or w values for all stations
    dec = np.radians(90-phase_target[0])
    ha = np.radians(phase_target[1]) 
    D = (1/wavelength) * np.array([Positions[0] - Positions[1],Positions[0] - Positions[2],Positions[1] - Positions[2]])
    matrix = np.array([[np.sin(ha),np.cos(ha),0],[-np.sin(dec)*np.cos(ha),np.sin(ha)*np.sin(dec),np.cos(dec)],[np.cos(dec)*np.cos(ha),-np.cos(dec)*np.sin(ha),np.sin(dec)]])
    uvw= np.matmul(D,np.transpose(matrix))
    u = np.append(uvw[0,:,0] , np.append(uvw[1,:,0] , uvw[2,:,0]))
    v = np.append(uvw[0,:,1] , np.append(uvw[1,:,1] , uvw[2,:,1]))
    w = np.append(uvw[0,:,2] , np.append(uvw[1,:,2] , uvw[2,:,2]))
    
    if psf_output == True:
        #create grid for psf 
        extrema = 1.01*np.min((u,v)),np.max((u,v))
        u_points = np.linspace(extrema[1],extrema[0],grid_resolution)
        v_points = np.linspace(extrema[0],extrema[1],grid_resolution)
        grid_u,grid_v = np.meshgrid(u_points,v_points)
        grid = np.zeros(np.shape(grid_u))
        for i in range(len(u)):
            u_closest = np.argmin(abs(u_points-u[i]))
            v_closest = np.argmin(abs(v_points-v[i]))
            grid[v_closest][u_closest] = 1
        
        psf = abs(np.fft.fftshift(np.fft.fft2(grid)))
    
        return uvw, np.array((u,v,w)), psf ,grid, np.array((u_points,v_points))
    else:    
        return uvw, np.array((u,v,w))

#sample functions
def sample_compl(Baselines,source,I,wavelength,orien1,orien2,full_output=True):
    #simple sampling function
    Vs = np.zeros(len(Baselines),dtype=np.complex64)
    for j in range(len(Baselines)):
        V = 0 + 0j
        for i in range(len(source)):
            if full_output==True:
                antenna_target1 = rot_transform(orien1, source[i])
                antenna_target2 = rot_transform(orien2, source[i])
                An = dipole_inter(antenna_target1[0],antenna_target2[0])
            else:
                An = 1
            a = np.sin(source[i][0])*np.cos(source[i][1]),np.sin(source[i][0])*np.sin(source[i][1]),np.cos(source[i][0])
            D = Baselines[j] / wavelength
            D_dot_sig = np.dot(D,a)
            Real = An*I[i]*np.cos(2*np.pi*D_dot_sig)
            Im = An*I[i]*np.sin(2*np.pi*D_dot_sig)
            V += Real + 1j*Im
        Vs[j] = V
    return Vs

def Visibility_sampling(Positions,eclipse,model_x,model_I,wavelength,orientations=None,Noise=None, system_noise_factor=0.1, bandwidth=None, antenna_length=5 , noise_seed=None, bars=False):
    #main sampling function
    #takes positions of the satelites as argument
    #takes eclipse array of the shape [3,baselines,directions] (true for visible from each baseline, false for not)
    #takes the direction arguments of the model as well as their intensitity
    #wavelength
    #takes list of added point sources in structure [x,y,z,Intensity (per sterradian)]
    #orientations can be fed into an array of equal shape as the positions, if done it takes those into account\
        #if no orientations are given it takes makes the directional sensitivity 100% in all directions
    #Noise can be added with [mean , rms] in the complex plane, then it returns/
        #both the sampled function with and without noise plus noise only
    
    #setting options
    if type(orientations) == type(None):
        orientation_option = False
    else:
        orientation_option = True
        orientation_ground = orientations[0]
        orientation_s1 = orientations[1]
        orientation_s2 = orientations[2]
        
    if Noise == None:
        Noise_option = False
    else:
        Noise_option = True
        
    if Noise_option == True:
        #takes a 2D guassian grab for noise (real and imaginary parts)
        #can set noise seed for testing 
        if type(noise_seed) != type(None):
            np.random.seed = noise_seed
        noise_grab = np.random.multivariate_normal([Noise[0],Noise[0]],Noise[1]**2*0.5*np.eye(2),size=(len(Positions)*len(Positions[0])))
        noise_sky = noise_grab[:,0] + 1j*noise_grab[:,1]
        noise_grab2 = np.random.multivariate_normal([0,0],system_noise_factor*Noise[1]**2*0.5*np.eye(2),size=(len(Positions)*len(Positions[0])))
        noise_sys = noise_grab2[:,0] + 1j*noise_grab2[:,1]
        noise = np.reshape(noise_sky+noise_sys,[len(Positions),len(Positions[0])])
    

    #initializing arrays
    #can take dtyope as complext 64 or 128, faster or more accurate 
    Vis_array = np.zeros((len(Positions),len(Positions[0])),dtype=np.complex128)
        #Vis_array_sources = np.zeros((len(Positions),len(Positions[0])),dtype=np.complex64)
    #getting Baselines
    Baselines = np.array([Positions[0] - Positions[1],Positions[0] - Positions[2],Positions[1] - Positions[2]])
    if bars == False:
        print('Adding Visibilities')
    #adding visibilities
    for i in tqdm(range(len(Positions[0])),disable=bars):
        for j in range(len(Positions)):
            D = Baselines[j][i] / wavelength
            D_dot_sig = np.matmul(D,np.transpose(model_x))
            if orientation_option ==False:
                An = 1 
            elif j == 0:
                An = dipole_An(orientation_ground[i],orientation_s1[i],model_x)
            elif j == 1:
                An = dipole_An(orientation_ground[i],orientation_s2[i],model_x) 
            elif j == 2:
                An = dipole_An(orientation_s1[i],orientation_s2[i],model_x) 
            
            Real = An*model_I*np.cos(2*np.pi*D_dot_sig) * eclipse[j][i]
            Im = An*model_I*np.sin(2*np.pi*D_dot_sig) * eclipse[j][i]
            Vis_array[j][i] = (np.sum(Real) + 1j*np.sum(Im))
    if Noise_option==False:
        return Vis_array, Baselines
    elif Noise_option==True:
        Vis_with_noise = Vis_array+noise
        return Vis_with_noise,Baselines,Vis_array, noise

def Visibility_sampling_weclipse(Positions, model_x,model_I,wavelength,orientations=None,Noise=None, system_noise_factor=0.1, bandwidth=None, antenna_length=5 ,measure_map_option=False, noise_seed=None, bars=False):
    #main sampling function
    #same as the normal one except this one calculates the eclipse per time stamp, uses less memory 
    #takes positions of the satelites as argument
    #takes the direction arguments of the model as well as their intensitity
    #wavelength
    #takes list of added point sources in structure [x,y,z,Intensity (per sterradian)]
    #orientations can be fed into an array of equal shape as the positions, if done it takes those into account\
        #if no orientations are given it takes makes the directional sensitivity 100% in all directions
    #Noise can be added with [mean , rms] in the complex plane, then it returns/
        #both the sampled function with and without noise plus noise only
    #can output a map with the amount of measurements each direction gets, 1 for how much a direction is visible for all station and 1 for at least one baseline
    #setting options
    if type(orientations) == type(None):
        orientation_option = False
    else:
        orientation_option = True
        orientation_ground = orientations[0]
        orientation_s1 = orientations[1]
        orientation_s2 = orientations[2]
        
    if Noise == None:
        Noise_option = False
    else:
        Noise_option = True
        
    if Noise_option == True:
        if type(noise_seed) != type(None):
            np.random.seed = noise_seed
        noise_grab = np.random.multivariate_normal([Noise[0],Noise[0]],Noise[1]**2*0.5*np.eye(2),size=(len(Positions)*len(Positions[0])))
        noise_sky = noise_grab[:,0] + 1j*noise_grab[:,1]
        noise_grab2 = np.random.multivariate_normal([0,0],system_noise_factor*Noise[1]**2*0.5*np.eye(2),size=(len(Positions)*len(Positions[0])))
        noise_sys = noise_grab2[:,0] + 1j*noise_grab2[:,1]
        noise = np.reshape(noise_sky+noise_sys,[len(Positions),len(Positions[0])])
        
    if measure_map_option == True:
        measurement_map = np.zeros([2,len(model_x)]) #counts when baselines see a direction, first row for when a single measurement, second row counts when all baselines a direction

    #initializing arrays
    Vis_array = np.zeros((len(Positions),len(Positions[0])),dtype=np.complex128)
        #Vis_array_sources = np.zeros((len(Positions),len(Positions[0])),dtype=np.complex64)
    #getting Baselines
    Positions = np.asarray(Positions) #to change from tuple to array for slicing
    Baselines = np.array([Positions[0] - Positions[1],Positions[0] - Positions[2],Positions[1] - Positions[2]])
    if bars == False:
        print('Adding Visibilities')
    #adding visibilities
    for i in tqdm(range(len(Positions[0])),disable=bars):
        eclipse = eclipse_checker_single(Positions[:,i,:], model_x)
        for j in range(len(Positions)):
            D = Baselines[j][i] / wavelength
            D_dot_sig = np.matmul(D,np.transpose(model_x))
            if orientation_option ==False:
                An = 1 
            elif j == 0:
                An = dipole_An(orientation_ground[i],orientation_s1[i],model_x)
            elif j == 1:
                An = dipole_An(orientation_ground[i],orientation_s2[i],model_x) 
            elif j == 2:
                An = dipole_An(orientation_s1[i],orientation_s2[i],model_x) 
            
            Real = An*model_I*np.cos(2*np.pi*D_dot_sig) * eclipse[j]
            Im = An*model_I*np.sin(2*np.pi*D_dot_sig) * eclipse[j]
            Vis_array[j][i] = (np.sum(Real) + 1j*np.sum(Im))
            
        if measure_map_option == True:
            measurement_map[0] += np.sum(eclipse,axis=0)
            measurement_map[1] += np.prod(eclipse,axis=0)
    if Noise_option==False and measure_map_option==False:
        return Vis_array, Baselines
    elif Noise_option==True and measure_map_option==False:
        Vis_with_noise = Vis_array+noise
        return Vis_with_noise,Baselines,Vis_array, noise
    elif Noise_option==False and measure_map_option==False:
        return Vis_array, Baselines, measurement_map
    elif Noise_option==True and measure_map_option==True:
        Vis_with_noise = Vis_array+noise
        return Vis_with_noise,Baselines, measurement_map, Vis_array, noise



def beachball(Baseline,x_hats,wavelength):
    #returns the phase structure of each direction (model_x) given the baseline
    #used in the reconstruction function to generate the map
    #essentially assignes a phase to each model_x direction that can be compared to the phase of the visibility
    a = np.transpose(x_hats)
    D = Baseline / wavelength
    D_dot_sig = np.matmul(D,a) #matmul is better than dot
    Real = np.cos(2*np.pi*D_dot_sig)
    Im = np.sin(2*np.pi*D_dot_sig)
    V = Real + 1j*Im
    phase = np.angle(V)
    return phase

def reconstruct(Positions, Vs, NPIX, model_x ,wavelength, divide_by_baselines=False,calibration=False, Vs_c=0, bars=False):
    #recconstructs the image from the visibilities
    #calibration can add a point for testing
    #divide by baselines divides the end result by the amount of baselines\
        #because the function basically adds all the baselines together
        # for real use should probably divide each direction by number of counts as visisble (addendum: not necicarily true this)
    Baselines = Positions[0]-Positions[1],Positions[0]-Positions[2], Positions[1]-Positions[2]
    ring_struc = np.zeros(NPIX)
    if calibration==True:
        ring_struc_calib = np.zeros(NPIX)
    t0 = time.time()
    angles = np.angle(Vs)
    I = abs(Vs)
    if bars == False:
        print('Reconstructing Visibilities')
    for j in tqdm(range(len(Baselines[0])),disable=bars):
        phase_ring1 = beachball(Baselines[0][j], model_x, wavelength)
        phase_angle1 = I[0][j]*np.cos(angles[0][j]-phase_ring1)
        phase_ring2 = beachball(Baselines[1][j], model_x, wavelength)
        phase_angle2 = I[1][j]*np.cos(angles[1][j]-phase_ring2)
        phase_ring3 = beachball(Baselines[2][j], model_x, wavelength)
        phase_angle3 = I[2][j]*np.cos(angles[2][j]-phase_ring3)
        ring_struc += phase_angle1 + phase_angle2 + phase_angle3
        if calibration==True:
            phase_ring1_c = beachball(Baselines[0][j], model_x, wavelength)
            phase_angle1_c = abs(Vs_c[0][j])*np.cos(np.angle(np.exp(1j*(np.angle(Vs_c[0][j])-phase_ring1_c))))
            phase_ring2_c = beachball(Baselines[1][j], model_x, wavelength)
            phase_angle2_c = abs(Vs_c[1][j])*np.cos(np.angle(np.exp(1j*(np.angle(Vs_c[1][j])-phase_ring2_c))))
            phase_ring3_c = beachball(Baselines[2][j], model_x, wavelength)
            phase_angle3_c = abs(Vs_c[2][j])*np.cos(np.angle(np.exp(1j*(np.angle(Vs_c[2][j])-phase_ring3_c))))
            ring_struc_calib += phase_angle1_c + phase_angle2_c + phase_angle3_c
    if divide_by_baselines == True:
        ring_struc = ring_struc /(len(Positions[0])*len(Positions))
    t_end = time.time()
    if bars == False:
        print('reconstruction took',round(t_end-t0,1),'seconds')
        print()
    if calibration==True:
        return ring_struc, ring_struc_calib
    else:
        return ring_struc
    
def reconstruct_tool(Ds, phase, Vs, NPIX):
    #tool to reconstruct the image given the arguments
    #mostly made to work with cleaning algorithm, should be more efficient than\
        #recontruct because it it doesnt have to recalculate repeated arrays
    #does not have calibration function
    #arguments are: Baselines divided by wavelength, the dot products of the Ds with the model_x\
        #the visibilities, the number of pixels in the image, bars wether or not to hide the progress bar
    image = np.zeros(NPIX)
    angles = np.angle(Vs)
    I = abs(Vs)
    #making the rings
    
    combination = np.cos(np.transpose(angles)-np.transpose(phase)) * np.transpose(I)
    image = abs(np.sum(np.sum(combination,axis=2),axis=1) / (len(Ds)*len(Ds[0])))
    return image

def stamp_reconstruction(target, size ,NSIDE ,Positions , Vs, wavelength,resolution=None):
    #input target direction in theta, phi, degrees 
    #input size of the stamp in degrees (up/down,left/right) this value represents the total size of the image
    #input the Positions of the stations
    #input the Visibilities
    t0=time.time()
    print('starting stamp reconstruction')
    Baselines = Positions[0]-Positions[1],Positions[0]-Positions[2], Positions[1]-Positions[2]
    arcmin = hp.nside2resol(NSIDE,arcmin=False) #checks what the resolution of the NSIDE map is
    the,ph = np.radians(target)
    area = np.radians(size[0])/2, np.radians(size[1])/2
    if resolution==None: #can be modified to change the number of pixels in a normal map
        pixels = 3*int(area[0]/arcmin), 3*int(area[1]/arcmin)
    else:
        pixels = resolution[0],resolution[1]
    theta = np.linspace(the-area[0],the+area[0],int(pixels[0]),endpoint=False)
    phi = np.linspace(ph-area[1],ph+area[1],int(pixels[1]),endpoint=False)
    THETA,PHI = np.meshgrid(theta,phi)
    grid = hp.pixelfunc.ang2vec(THETA,PHI)
    
    angles = np.angle(Vs) #gets the phase of the visibilities
    I = abs(Vs) #gets the absolute values of the visibility
    reconstruction = np.zeros((pixels)) #initialize an array to store the results
    for i in tqdm(range(len(Baselines[0]))):
        D1 = Baselines[0][i] / wavelength
        D1_dot_sig = np.matmul(grid,D1)
        phase1 = np.angle(np.cos(2*np.pi*D1_dot_sig)+1j*np.sin(2*np.pi*D1_dot_sig))
        D2 = Baselines[1][i] / wavelength
        D2_dot_sig = np.matmul(grid,D2)
        phase2 = np.angle(np.cos(2*np.pi*D2_dot_sig)+1j*np.sin(2*np.pi*D2_dot_sig))
        D3 = Baselines[2][i] / wavelength
        D3_dot_sig = np.matmul(grid,D3)
        phase3 = np.angle(np.cos(2*np.pi*D3_dot_sig)+1j*np.sin(2*np.pi*D3_dot_sig))
        
        contr1 = I[0][i]*np.cos(angles[0][i]-phase1)
        contr2 = I[1][i]*np.cos(angles[1][i]-phase2)
        contr3 = I[2][i]*np.cos(angles[2][i]-phase3)
        
        reconstruction += contr1 + contr2 + contr3
        
    tend = time.time()
    print('time elapsed for stamp reconstruction is',round(tend-t0,1),'seconds')
    return reconstruction, [theta,phi]

def grid_testing(target,size,Positions,wavelength,resolution=(200,200), source_amount=5, source_strength = 'random',seed=None):
    Baselines = np.array([Positions[0]-Positions[1],Positions[0]-Positions[2], Positions[1]-Positions[2]])
    D = Baselines/wavelength
    if seed != None:
        np.random.seed(seed)
    if source_amount==1:
        sources = np.radians(target)
        x_hats = hp.pixelfunc.ang2vec(sources[0], sources[1])
        source_strength = 1

    elif source_amount != 1:
        sources = np.radians(np.random.rand(source_amount,2)*(np.array(size)/2) + target)
        x_hats=hp.pixelfunc.ang2vec(sources[:,0], sources[:,1])
        if source_strength =='random':
            source_strength = 5*np.random.rand(source_amount)
    
    the,ph = np.radians(target)
    area = np.radians(size[0])/2, np.radians(size[1])/2
    pixels = resolution
    theta = np.linspace(the-area[0],the+area[0],int(pixels[0]),endpoint=False)
    phi = np.linspace(ph-area[1],ph+area[1],int(pixels[1]),endpoint=False)
    THETA,PHI = np.meshgrid(theta,phi)
    grid = hp.pixelfunc.ang2vec(THETA,PHI)
    if source_amount !=1:
        sources_grid = np.zeros((source_amount,2))
        source_map = np.zeros((resolution))
        for k in range(len(sources)):
            theta_s = int(np.argmin(abs(theta-sources[k,0])))
            phi_s = int(np.argmin(abs(phi-sources[k,1])))
            sources_grid[k][0] = theta_s
            sources_grid[k][1] = phi_s
            source_map[theta_s,phi_s] = source_strength
        
    
    Vs = np.zeros((3,len(Positions[0])),dtype=np.complex128)
    for j in range(len(Positions[0])):
        D_dot_sig_1 = np.matmul(D[0][j],np.transpose(x_hats))
        D_dot_sig_2 = np.matmul(D[1][j],np.transpose(x_hats))
        D_dot_sig_3 = np.matmul(D[2][j],np.transpose(x_hats))
        Vs[0][j] = (np.sum(source_strength*(np.cos(2*np.pi*D_dot_sig_1) + 1j*np.sin(2*np.pi*D_dot_sig_1))))
        Vs[1][j] = (np.sum(source_strength*(np.cos(2*np.pi*D_dot_sig_2) + 1j*np.sin(2*np.pi*D_dot_sig_2))))
        Vs[2][j] = (np.sum(source_strength*(np.cos(2*np.pi*D_dot_sig_3) + 1j*np.sin(2*np.pi*D_dot_sig_3))))
                
    I=abs(Vs)
    angles = np.angle(Vs)
    reconstruction = np.zeros((pixels))
    for i in tqdm(range(len(Baselines[0]))):
        D1_dot_sig = np.matmul(grid,D[0][i])
        phase1 = np.angle(np.cos(2*np.pi*D1_dot_sig)+1j*np.sin(2*np.pi*D1_dot_sig))
        D2_dot_sig = np.matmul(grid,D[1][i])
        phase2 = np.angle(np.cos(2*np.pi*D2_dot_sig)+1j*np.sin(2*np.pi*D2_dot_sig))
        D3_dot_sig = np.matmul(grid,D[2][i])
        phase3 = np.angle(np.cos(2*np.pi*D3_dot_sig)+1j*np.sin(2*np.pi*D3_dot_sig))
        
        contr1 = I[0][i]*np.cos(angles[0][i]-phase1)
        contr2 = I[1][i]*np.cos(angles[1][i]-phase2)
        contr3 = I[2][i]*np.cos(angles[2][i]-phase3)
        
        reconstruction += contr1 + contr2 + contr3
    if source_amount !=1:
        return reconstruction, source_map ,np.degrees([theta,phi])
    if source_amount == 1:
        return reconstruction, np.degrees([theta,phi])


def Cleaning(Vs,eclipse, Baselines, model_x, wavelength, floor = 10, cycles=None, points_per_cycle=20, break_condition = 100):
    #takes the Visibilities and after reconstruction takes away the points_per_cycle
    #clearest points
    #does this an amount of times equal to cycles if given, else until either floor or break condition is reached
    #returns an array with the visibilities taken away for each cycle 
    #also a final array with the cumulative effect of all cycles
    #finally an array with the coordinates and strength of the points taken away per cycle
    #inititate timer
    #this one encounters a bug in the multithreading of python 3.8, use the other one at the bottom if you are using that version
    print('initializing cleaning algorithm')
    print('')
    t0 = time.time()
    floor_reached = True
    if type(cycles) !=None:
        break_condition = cycles 
    #intitiate the arrays to be filled
    Vs_clean_tot = np.zeros((3,len(Vs[0])),dtype=np.complex128)
    #get the model parameters
    NPIX = len(model_x)
    NSIDE = hp.npix2nside(NPIX)
    #initiate the arrays that tell what the function did
    points_removed = np.array([])
    pixels = np.array([])
    full_sky = np.array([])
    Vs_cleaning = np.array([])
    #get resources that dont need to be recalculated every loop
    print('generating reused parameters \n this might take a bit of time')
    print('')
    D = Baselines / wavelength
    D_dot_sig = np.matmul(D,np.transpose(model_x))
    make_compl = 2j*np.pi
    #comp = np.exp(2j*np.pi*D_dot_sig) #was super slow the evaluate function was a factor 4 times as fast
    comp = ne.evaluate('exp(make_compl*D_dot_sig)')
    del D_dot_sig #deleting some variables to clear memory
    del Baselines
    del make_compl
    phase = np.angle(comp)
    eclipsed_phase_array = comp * eclipse
    del eclipse
    print('starting Cleaning algorithm')
    I_max = 1e7
    i=0
    I_max_old = 1e10
    pbar = tqdm(total=break_condition) #create progress bar
    while I_max >= floor and i<break_condition:
        Vs_active = Vs - Vs_clean_tot
        sky = reconstruct_tool(D,phase,Vs_active,NPIX) #make reconstruction
        max_points = (-abs(sky)).argsort()[:points_per_cycle] #identify brightests points
        I_max = np.max(abs(sky)) #identify brightest
        if I_max_old < I_max: #create a break condition
            print('non-convergence after',i,'cycles')
            floor_reached = False
            break
        full_sky = np.append(full_sky,sky) 
        pixels = np.append(pixels,max_points)
        I = sky[max_points] *0.2
        empty_sky = np.zeros(len(model_x))
        empty_sky[max_points] = I
        Visabilities_cleaning = np.sum(empty_sky * eclipsed_phase_array,axis=2)
        Vs_cleaning = np.append(Vs_cleaning,Visabilities_cleaning)
        Vs_clean_tot += Visabilities_cleaning
        vectors_removed = np.transpose(hp.pixelfunc.pix2vec(NSIDE, max_points))
        full_removed = np.column_stack((vectors_removed,I))
        points_removed = np.append(points_removed,np.ndarray.flatten(full_removed))
        I_max_old = I_max
        i+=1
        pbar.update(1) #update progress bar
        if i%5 == 0:
            print('Maximum intensity found is',I_max,)
    
    pbar.close() #close progress bar
    points_removed = np.reshape(points_removed,(i,points_per_cycle,4))
    pixels = np.reshape(pixels,(i,points_per_cycle))
    full_sky = np.reshape(full_sky,(i,len(model_x)))
    Vs_cleaning=np.reshape(Vs_cleaning,(i,3,len(Vs[0])))
    tend = time.time()
    if i < break_condition and floor_reached==True:
        print('floor reached after',i,'itterations')
        print('')
    print('"Clean" operation took',round(tend-t0,1),'seconds')
    print('operation took',i,'loops')
    return full_sky, Vs_clean_tot,Vs_cleaning, points_removed, pixels

def gaussian_blob(max_intensity,model_x,direction,size_ang,integrated=False):
    #function to add a gaussian blob that can be added to healpix map
    #max_intensity is the intensity at tghe center, it falls off with sigma=size_ang (radians)
    #if integrated = True it uses the max intensity as the total integrated value for the source over the arc
    NSIDE = hp.npix2nside(len(model_x))
    direction_x = hp.ang2vec(direction[0], direction[1])
    pixels = hp.query_disc(NSIDE,direction_x,size_ang)
    inner_products = np.dot(model_x[pixels],direction_x)
    inner_product_range = abs(np.max(inner_products)-np.min(inner_products))
    values = max_intensity*np.exp(-((inner_products-1)/(0.5*inner_product_range))**2)
    if integrated == True:
        values = values/np.sum(values) *max_intensity
    model_I_blob = np.zeros(len(model_x))
    model_I_blob[pixels] += values
    return model_I_blob

def make_model(NSIDE,frequency,object_intensity=None,object_size=None,object_direction=[0,0],integrated=False):
    #function to make a pygdsm model at a certain frequency (minimum is 10 MHz)
    #possibility to add objects via the object_intensity and object size/direction arguments
    #returns the model map at NSIDE specified and the the direction (unit) vectors 
    #Model is given in jansky per point 
    
    #looks at arguments
    if type(object_intensity) == type(None):
        add_object=False
    else:
        add_object = True
    if type(object_size) != type(None):
        gaussian = True
    else:
        gaussian = False
    if type(object_direction[0]) == type(np.array([])):
        multiple_sources = True
    else:
        multiple_sources = False
    
    NPIX = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(nside=NSIDE,ipix=np.arange(0,NPIX))
    x,y,z = hp.pix2vec(nside=NSIDE,ipix=np.arange(0,NPIX))
    model_x = np.column_stack((x,np.column_stack((y,z))))
    gdsm2016 = pygdsm.GlobalSkyModel2016(freq_unit='MHz' ,data_unit='MJysr')
    gdsm2016.generate(frequency/1e6)
    model_I = abs(hp.pixelfunc.ud_grade(gdsm2016.generate(frequency/1e6),NSIDE))
    model_I_jansky = model_I * hp.pixelfunc.nside2pixarea(NSIDE, degrees=False) * 1e6 #converts from MJ/sr to jansky per pixel
    
    if add_object == True and multiple_sources == True and gaussian == True:
        for i in range(len(object_direction)):
            modification = gaussian_blob(object_intensity[i], model_x, object_direction[i], object_size[i],integrated=integrated)
            model_I_jansky += modification
    elif add_object == True and multiple_sources == True and gaussian == False:
        pixels = hp.ang2pix(NSIDE, object_direction[:,0], object_direction[:,1])
        model_I_jansky[pixels] += object_intensity 
    elif add_object == True and multiple_sources == False and gaussian == True:
        modification = gaussian_blob(object_intensity, model_x, object_direction, object_size,integrated=integrated)
        model_I_jansky += modification
    elif add_object == True and multiple_sources == False and gaussian == False:
        pixel = hp.ang2pix(NSIDE, object_direction[0], object_direction[1])
        model_I_jansky[pixel] += object_intensity
    
    return model_I_jansky, model_x



def source_invasion(Positions,eclipse,model_x,source_positions,source_size,source_strength,Vs,wavelength,NSIDE):
    #function to add Visibilities of a moving source in a visibility set
    #uses source_positions in [x,y,z] format for every point (unit vectors)
    D = 1/wavelength * np.array([Positions[0] - Positions[1],Positions[0] - Positions[2],Positions[1] - Positions[2]])
    An=1
    V = np.zeros(np.shape(Vs),dtype=np.complex128)
    for i in tqdm(range(len(source_positions))):
        pixels = hp.query_disc(NSIDE,source_positions[i],source_size)
        relevant_x = model_x[pixels]
        relevant_eclipse = eclipse[:,i,pixels]
        D_dot_sig1 = np.matmul(D[0][i],np.transpose(relevant_x))
        D_dot_sig2 = np.matmul(D[1][i],np.transpose(relevant_x)) 
        D_dot_sig3 = np.matmul(D[2][i],np.transpose(relevant_x))
        V1 = An*source_strength*np.exp(2j*np.pi*D_dot_sig1) * relevant_eclipse[0]*relevant_eclipse[1]
        V2 = An*source_strength*np.exp(2j*np.pi*D_dot_sig2) * relevant_eclipse[0]*relevant_eclipse[2]
        V3 = An*source_strength*np.exp(2j*np.pi*D_dot_sig3) * relevant_eclipse[1]*relevant_eclipse[2]
        V[0][i] = np.sum(V1) 
        V[1][i] = np.sum(V2)
        V[2][i] = np.sum(V3)
    
    modified_V = Vs+V
    return modified_V, V


def Cleaning_2(Vs, Positions, model_x, wavelength, floor = 10, flat_factor=0.2, cycles=None, points_per_cycle=20, break_condition = 100):
    #takes the Visibilities and after reconstruction takes away the points_per_cycle
    #clearest points
    #does this an amount of times equal to cycles if given, else until either floor or break condition is reached
    #returns an array with the visibilities taken away for each cycle 
    #also a final array with the cumulative effect of all cycles
    #finally an array with the coordinates and strength of the points taken away per cycle
    #inititate timer
    #same as the other cleaning except it circumvents a bug in python 3.8, but is less efficient 
    print('initializing cleaning algorithm')
    print('')
    t0 = time.time()
    floor_reached = True
    if type(cycles) !=None:
        break_condition = cycles 
    #intitiate the arrays to be filled
    Vs_clean_tot = np.zeros((3,len(Vs[0])),dtype=np.complex128)
    #get the model parameters
    NPIX = len(model_x)
    NSIDE = hp.npix2nside(NPIX)
    #initiate the arrays that tell what the function did
    points_removed = np.array([])
    pixels = np.array([])
    full_sky = np.array([])
    Vs_cleaning = np.array([])
    print('starting Cleaning algorithm')
    I_max = 1e7
    i=0
    I_max_old = 1e10
    pbar = tqdm(total=break_condition) #create progress bar
    while I_max >= floor and i<break_condition:
        sky = reconstruct(Positions, Vs-Vs_clean_tot, NPIX, model_x, wavelength, divide_by_baselines=True ,bars=True)
        max_points = (-abs(sky)).argsort()[:points_per_cycle] #identify brightests points
        I_max = np.max(abs(sky)) #identify brightest
        if I_max_old < I_max: #create a break condition
            print('non-convergence after',i,'cycles')
            floor_reached = False
            break
        full_sky = np.append(full_sky,sky) 
        pixels = np.append(pixels,max_points)
        I = sky[max_points] * flat_factor
        empty_sky = np.zeros(len(model_x))
        empty_sky[max_points] = I
        Visabilities_cleaning = Visibility_sampling_weclipse(Positions, model_x, empty_sky, wavelength, bars=True)[0]
        Vs_cleaning = np.append(Vs_cleaning,Visabilities_cleaning)
        Vs_clean_tot += Visabilities_cleaning
        vectors_removed = np.transpose(hp.pixelfunc.pix2vec(NSIDE, max_points))
        full_removed = np.column_stack((vectors_removed,I))
        points_removed = np.append(points_removed,np.ndarray.flatten(full_removed))
        I_max_old = I_max
        i+=1
        pbar.update(1) #update progress bar
        if i%5 == 0:
            print('Maximum intensity found is',I_max,)
    
    pbar.close() #close progress bar
    points_removed = np.reshape(points_removed,(i,points_per_cycle,4))
    pixels = np.reshape(pixels,(i,points_per_cycle))
    full_sky = np.reshape(full_sky,(i,len(model_x)))
    Vs_cleaning=np.reshape(Vs_cleaning,(i,3,len(Vs[0])))
    tend = time.time()
    if i < break_condition and floor_reached==True:
        print('floor reached after',i,'itterations')
        print('')
    print('"Clean" operation took',round(tend-t0,1),'seconds')
    print('operation took',i,'loops')
    return full_sky, Vs_clean_tot,Vs_cleaning, points_removed, pixels







