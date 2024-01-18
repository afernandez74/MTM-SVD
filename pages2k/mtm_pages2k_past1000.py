#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:18:06 2023

@author: afer
"""

from mtm_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle 
import time

#%% load the pages2k compiled data

path = os.path.expanduser("~/mtm_local/pages2k/past1000_comp_dic/")
files = os.listdir(path)
file = files[0] 

with open(path+file, 'rb') as f:
    pages2k_dic = pickle.load(f) 


# %% 2) Compute LFV spectra for all simulations individually


# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 


# dictionary to save results
pages2k_past1000_mtm_svd_results={}

for key,value in pages2k_dic.items():
    if key.startswith('values_normalized'):
        
        # no weights
        # [xx,yy] = np.meshgrid(ds_i.lon.values,ds_i.lat.values)
        # w = np.sqrt(np.cos(np.radians(yy)));
        # w = w.reshape(1,w.shape[0]*w.shape[1],order='F')
        # reshape 'tas' matrix to 2d
        if key == 'values_normalized':
            archive = 'ALL'
        else:
            archive = key[len('values_normalized_'):]
                
        tas2d = np.transpose(np.array(value)) # rows are time and columns space
        w = np.ones((1,tas2d.shape[1]))
        # calculate the LFV
        print(f"Calculating LFV spectrum for {archive}")
        freq, lfv = mtm_svd_lfv(tas2d, nw, kk, dt, w)
        
        # Assign results to variables
        freq_key = f"{archive}_freq"
        freq_value = freq
        lfv_key = f"{archive}_lfv"
        lfv_value = lfv
        
        # save to dictionary
        pages2k_past1000_mtm_svd_results[freq_key] =  freq
        pages2k_past1000_mtm_svd_results[lfv_key] = lfv
del key, value    

# save results to dic  
path = os.path.expanduser('~/mtm_local/pages2k/mtm_results/')
file_name = 'pages2k_past1000_mtm_results'
dic = pages2k_past1000_mtm_svd_results
with open(path + file_name, 'wb') as f:
    pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)
    
#%% confidence intervals
# Values for Confidence Interval calculation

ds = pages2k_dic['values_normalized']
niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.5] # confidence levels

tas2d = np.transpose(np.array(ds)) # rows are time and columns space
w = np.ones((1,tas2d.shape[1]))

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
freq,lfv = mtm_svd_lfv(tas2d, nw, kk, dt, w)
[conffreq, conflevels] = mtm_svd_conf(tas2d,nw,kk,dt,niter,sl,w) 

# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly

mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)
lfv_mean = np.mean(lfv)
adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values

path = os.path.expanduser('~/mtm_local/pages2k/mtm_results/')
file_name = 'pages2k_past1000_mtm_results_ci'
with open(path + file_name, 'wb') as f:
    pickle.dump(adj_ci, f, protocol=pickle.HIGHEST_PROTOCOL)
