# Script for MTM_SVD analysis of the CESM Last Millennium Ensemble
# the script analyzes all of the 13 members of the ensemble, produces LFV spectra 
# for both forced+internal and internal-only series, and produces figures combining
# all results

# %% import functions and packages

from mtm_funcs import *
from readin_funcs_CESM_LME import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl
import time

# %%-----------------
# 1) Load the dictionary with annualized simulation data
# -------------------

# file path where dictionary was saved in mtm_preprocessing.py script
file_path = os.path.expanduser("~/mtm_local/CESM_LME_data_dic/")
# obtain name of file (only works if one file is present only)
file = listdir(file_path)[0]
# save dictionary to dictionary variable
with open(file_path + file, 'rb') as f:
    CESM_LME_dic = pkl.load(f)

# Key values from the dictionary, which are equivalent to the number of the simulations (001-013)
sims = list(CESM_LME_dic.keys())

# %%-----------------
# 3) Compute the first ensemble member's LFV spectrum and confidence intervals
# -------------------

# start a timer
start_time = time.time()

# assign a reference simulation (CESM LME ensemble member 001) to which other analyses
# will be compared
sim_ref = CESM_LME_dic[sims[0]]

# reference temperature data
tas_ref = sim_ref.tas

# years of data
years = sim_ref.time

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

# Weights based on latitude
[xx,yy] = np.meshgrid(sim_ref.lon,sim_ref.lat)
w = np.sqrt(np.cos(np.radians(yy)));

# Analize all data up to 1850
cutoff_yr = 1850

# =============================================================================
# LFV calculation
# =============================================================================

# delete data post cutoff year
ix_post_cutoff = np.where(years>cutoff_yr)[0]
tas_ref = np.delete(tas_ref,(ix_post_cutoff),axis = 0)
years = np.delete(years,(ix_post_cutoff))

# reshape data to 2d
tas_ref = reshape_3d_to_2d(tas_ref)
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# calculate the LFV 
freq_ref, lfv_ref = mtm_svd_lfv(tas_ref, nw, kk, dt, w)

# end timer
end_time = time.time()

# elapsed time of calculation 
run_time = end_time - start_time
print (f'Run time of calculation: {run_time:.3f} seconds')


# %%-----------------
# 4) Compute the confidece intervals for the reference data 
# -------------------

# start a timer
start_time = time.time()

# =============================================================================
# Values for Confidence Interval calculation
# =============================================================================
niter = 10    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
[conffreq, conflevels] = mtm_svd_conf(tas_ref,nw,kk,dt,niter,sl,w) 

# =============================================================================
# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly
# =============================================================================

# Rescaling of confidence intervals 
fr_sec = nw/(tas_ref.shape[0]*dt) # secular frequency value
fr_sec_ix = np.where(freq_ref < fr_sec)[0][-1] 

lfv_mean = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 
mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values

# end timer
end_time = time.time()

# elapsed time of calculation 
run_time = end_time - start_time
print (f'Run time of confidence interval calculation: {run_time/60:.2f} minutes')

# %%-----------------
# 5) Compute the ensemble mean and internal-variability-only data (i.e., forcing series removed)
# -------------------

# start a timer
start_time = time.time()

# array where all data will be stored for calculation of the ensemble mean
tas_all = np.zeros((np.shape(tas_ref)[0],np.shape(tas_ref)[1],len(CESM_LME_dic)))
c=0

# loop that reads in each key,value pair from the dictionary, deletes post-industrial data
# reshapes it to 2d and then saves the 2d data in a tas_all array which contains all 13
# 2d arrays for each simulation
for key,value in CESM_LME_dic.items():
    tas_i = value.tas #obtain current simulation temperature data
    tas_i = np.delete(tas_i,(ix_post_cutoff),axis = 0) # delete post-industrial data
    tas_i = reshape_3d_to_2d(tas_i) # reshape data to 2d
    tas_all[:,:,c] = tas_i # assign to tas_all array 
    c=c+1
    del tas_i
del c, key, value

tas_ens_mn = np.nanmean(tas_all,axis = 2) # calculate ensemble mean gridded data
tas_ens_mn_glob = np.nanmean(tas_ens_mn,axis = 1) # calculate ensemble mean timeseries 

# =============================================================================
# Subtract ensemble mean from each simulation data
# =============================================================================

# tas_inter --> forcing removed / internal-only data 
# subtract ensemble mean from each field to obtain the internal-only fields
tas_inter = tas_all - tas_ens_mn[:,:,np.newaxis]
    
# end timer
end_time = time.time()

# elapsed time of calculation 
run_time = end_time - start_time
print (f'Run time of LFV spectra for CESM LME _all calculation: {run_time:.0f} seconds')


# %%-----------------
# 6) Compute the LVF spectra of the internal+forced ensemble members
# -------------------

# start a timer
start_time = time.time()

# Initialize LFV matrix to store all results 
lfv_all = np.zeros((lfv_ref.shape[0],tas_all.shape[2]))
for i in range(0,tas_all.shape[2]):
    tas =  tas_all[:,:,i]    
    # Compute the LFV    
    [freq, lfv] = mtm_svd_lfv(tas,nw,kk,dt,w)
    print(f'Calculating LFV for "all" data of simulation {i+1:03d}')
    lfv_all[:,i] = lfv
del freq, lfv, i, tas

# end timer
end_time = time.time()

# elapsed time of calculation 
run_time = end_time - start_time
print (f'Run time of LFV spectra for CESM LME _internal calculation: {run_time:.0f} seconds')

# %%-----------------
# 7) Compute the LFV spectra of the internal-only ensemble members
# -------------------

# start a timer
start_time = time.time()

# Initialize LFV matrix to store all results 
lfv_inter = np.zeros((lfv_ref.shape[0],tas_inter.shape[2]))
for i in range(0,tas_inter.shape[2]):
    tas =  tas_inter[:,:,i]    
    # Compute the LFV    
    [freq, lfv] = mtm_svd_lfv(tas,nw,kk,dt,w)
    print(f'Calculating LFV for "internal-only" data of simulation {i+1:03d}')
    lfv_inter[:,i] = lfv
del freq, lfv, i, tas

# end timer
end_time = time.time()

# elapsed time of calculation 
run_time = end_time - start_time
print (f'Run time of LFV spectra for CESM LME _internal calculation: {run_time:.0f} seconds')

# %%-----------------
# 6) Save stuff
# -------------------

# add results into a dictionary
CESM_LME_mtm_svd_results = {
    'data_source': file_path+file,
    'sim_ref': sim_ref.name,
    'lfv_ref': lfv_ref,
    'freq_ref': freq_ref,
    'sl': sl,
    'niter': niter,
    'conflevels': conflevels,
    'conflevels_adjusted': adj_ci,
    'ta_ensemble_mean_global': tas_ens_mn_glob,
    'lfv_all': lfv_all,
    'lfv_inter': lfv_inter
    }
results_save_path = os.path.expanduser("~/mtm_local/CESM_LME_mtm_svd_results/")
results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CESM_LME_mtm_svd_results{results_timestamp}'
results_full_path = results_save_path+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(CESM_LME_mtm_svd_results, f)


