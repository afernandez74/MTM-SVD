# Script for MTM_SVD analysis of the CMIP6 past1000 experiments
# the script analyzes all 4 members of the ensemble, produces LFV spectra 
# for both forced+internal and internal-only(unforced) series

# %% import functions and packages

from mtm_funcs import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import time

# %% 1) Load data and modify dictionary
#imported from Mann et al.,2021 data 
# original MATLAB data imported with "mat_to_py.py" script

# file path where dictionary was saved in mtm_preprocessing.py script
file_path = os.path.expanduser("~/mtm_local/cmip5_past1000/datasets/")

# obtain name of file 
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

# model names
models = [file_name.replace('_past1000_dataset.nc', '') for file_name in files]

# standardize time bounds for all analyses (remove first and last years for irregularities)
year_i = 851
year_f = 1849

# create dictionary with all standardized datasets
cmip5_past1000={}

for file in files:
    
    # load dataset 
    ds_i = xr.open_dataset(file_path+file)
    
    # get model name in file
    model = ds_i.attrs['model_name']
    
    # assign model data to dictionary, selecting only desired years
    cmip5_past1000[model] = ds_i.sel(time=slice(year_i,year_f))
        

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/tas_dic/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'cmip5_past1000_dic_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000, f)
print(f"results dictionary saved to {path_results}")
del f, ds_i, results_timestamp, results_file_name, results_full_path, 
del model, files, file, path_results, year_i, year_f

# %% 2) Compute LFV spectra 
#for all models

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

# =============================================================================
# Loop for MTM-SVD calculations
# =============================================================================

# dictionary to save results
cmip5_past1000_mtm_svd_results={}

for key,value in cmip5_past1000.items():
    ds_i = value
    model = key
    lat = ds_i['lat'].values
    lon = ds_i['lon'].values
    year = ds_i['time'].values
    tas_3d = ds_i['tas'].values
    
    # calculate weights based on latitude
    [xx,yy] = np.meshgrid(lon,lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # reshape 'tas' matrix to 2d and get rid of years outside range
    tas_2d = reshape_3d_to_2d(tas_3d)
    
    # calculate the LFV
    print(f"Calculating LFV spectrum for {model}")
    freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
    
    # Assign results to variables
    freq_key = f"{model}_freq"
    freq_value = freq
    lfv_key = f"{model}_lfv"
    lfv_value = lfv
    
    # save to dictionary
    cmip5_past1000_mtm_svd_results[freq_key] = freq
    cmip5_past1000_mtm_svd_results[lfv_key] = lfv
del key, value, lfv_key, lfv_value, freq_key, freq_value, ds_i, model, lat, lon, year, tas_3d, tas_2d, xx, yy, w, lfv, freq

# save results to dic  
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/mtm_svd_results/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CMIP5_past1000_mtm_svd_results{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_mtm_svd_results, f)

print(f"results dictionary saved to {path_results}")
del path_results, results_timestamp, results_file_name, results_full_path, f
# %% 3) regrid all data to common resolution

# Show grid sizes for all models 
print("Lat x Lon values for all models:")
for key, value in cmip5_past1000.items():
    lat = len(value['lat'])
    lon = len(value['lon'])
    print(f"resolution_{key} = {lat}x{lon}")
    
del key, value

# regrid all entries to CESM grid
# drop FGOALS_gl due to it being the only one not from 850-1850
# drop GISS_E2_R_2 since it's the one with an excursion in the middle of the run (see https://data.giss.nasa.gov/modelE/cmip5/)
cmip5_past1000_regrid = {}
ref = cmip5_past1000['GISS_E2_R_1']
for key, value in cmip5_past1000.items():
    if key != 'FGOALS_gl':
        ds_i = value
        ds_i_regridded = ds_i.interp_like(ref, method='linear')
        cmip5_past1000_regrid[key]=ds_i_regridded
print("___")
print("Lat x Lon values for regridded models:")

# Show new grid sizes 
for key, value in cmip5_past1000_regrid.items():
    lat = len(value['lat'])
    lon = len(value['lon'])
    print(f"resolution_{key} = {lat}x{lon}")
del key, value  

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/tas_dic_regrid/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_regrid, f)
    
print(f"results dictionary saved to {path_results}")

del ds_i, ds_i_regridded, f, lat, lon, path_results, results_file_name, results_full_path, results_timestamp

# %% 4) Compute the ensemble mean and unforced data (i.e., forcing series removed)

# drop all but one of the GISS members to avoid model bias
drop = ['GISS_E2_R_2',
        'GISS_E2_R_3',
        'GISS_E2_R_4',
        'GISS_E2_R_5',
        'GISS_E2_R_6',
        'GISS_E2_R_7',
        'GISS_E2_R_8',
        'GISS_E2_R_9']

# concatenate datasets to perform calculations without including all but one GISS iteration
cmip5_past1000_regrid_concat = {key: value for key, value in cmip5_past1000_regrid.items() if key not in drop}
cmip5_past1000_regrid_concat = xr.concat(list(cmip5_past1000_regrid.values()), dim='model', coords = 'minimal')

# =============================================================================
# # calculate ensemble mean 
# =============================================================================
cmip5_past1000_ensemb_mean = cmip5_past1000_regrid_concat.mean(dim='model')

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/tas_dic_regrid_ensemb_mean/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_ensemb_mean_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_ensemb_mean, f)

del path_results, results_timestamp, results_file_name, results_full_path, drop

# =============================================================================
# calculate unforced data (forced - ensemble mean)
# =============================================================================
cmip5_past1000_regrid_unforced = {}
#subtract ensemble mean from each individual dataarray
for key, value in cmip5_past1000_regrid.items():
    cmip5_past1000_regrid_unforced[key] = value - cmip5_past1000_ensemb_mean
    

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/tas_dic_regrid_unforced/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_unforced_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_regrid_unforced, f)

del path_results, results_file_name, results_full_path, results_timestamp, f

# %% 5) Compute the LVF spectra for 'unforced' datasets

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

# =============================================================================
# Loop for MTM-SVD calculations
# =============================================================================

# dictionary to save results
cmip5_past1000_unforced_mtm_svd_results={}

for key, value in cmip5_past1000_regrid_unforced.items():
    
    ds_i = value
    model = key
    lat = ds_i['lat'].values
    lon = ds_i['lon'].values
    year = ds_i['time'].values
    tas_3d = ds_i['tas'].values
    
    # calculate weights based on latitude
    [xx,yy] = np.meshgrid(lon,lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # reshape 'tas' matrix to 2d and get rid of years outside range
    tas_2d = reshape_3d_to_2d(tas_3d)
    

    # calculate the LFV
    print(f"Calculating LFV spectrum for {model}")
    freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
    
    # Assign results to variables
    freq_key = f"{model}_freq"
    freq_value = freq
    lfv_key = f"{model}_lfv"
    lfv_value = lfv
    
    # save to dictionary
    cmip5_past1000_unforced_mtm_svd_results[freq_key] =  freq
    cmip5_past1000_unforced_mtm_svd_results[lfv_key] = lfv
del key, value    

# save results to dic  
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/mtm_svd_unforced_results/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CMIP5_past1000_mtm_svd_results{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_unforced_mtm_svd_results, f)
    
print(f"results dictionary saved to {path_results}")
    


#%% 6) Calculate confidence intervals for one of the GISS simulations

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

ref_model = 'GISS_E2_R_1'
ds = cmip5_past1000[ref_model]
lat = ds['lat'].values
lon = ds['lon'].values
year = ds['time'].values
tas_3d = ds['tas'].values

# calculate weights based on latitude
[xx,yy] = np.meshgrid(lon,lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# reshape 'tas' matrix to 2d and get rid of years outside range
tas_2d = reshape_3d_to_2d(tas_3d)

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
[conffreq, conflevels] = mtm_svd_conf(tas_2d,nw,kk,dt,niter,sl,w) 
freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w) # run a single iteration for normalization

# =============================================================================
# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly
# =============================================================================

# Rescaling of confidence intervals 
fr_sec = nw/(tas_2d.shape[0]*dt) # secular frequency value
fr_sec_ix = np.where(conffreq < fr_sec)[0][-1] 

lfv_mean = np.nanmean(lfv[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 
mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values

# %% 7) Save stuff
# -------------------

# add results into a dictionary
cmip5_past1000_mtm_svd_results = {
    'data_source': 'past1000 CMIP5 data Mann et al., 2021',
    'sim_ref': ref_model,
    'lfv_ref': lfv,
    'freq_ref': freq,
    'sl': sl,
    'niter': niter,
    'conflevels': conflevels,
    'conflevels_adjusted': adj_ci,
    'lfv_all': cmip5_past1000_mtm_svd_results,
    'lfv_inter': cmip5_past1000_unforced_mtm_svd_results,
    'cmip5_past1000_dic': cmip5_past1000,
    'cmip5_past1000_inter': cmip5_past1000_regrid_unforced,
    }
results_save_path = os.path.expanduser("~/mtm_local/cmip5_past1000/mtm_svd_results_ALL/")
results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'cmip5_past1000_mtm_svd_results{results_timestamp}'
results_full_path = results_save_path+results_file_name

with open(results_full_path, 'wb') as f:
    pickle.dump(cmip5_past1000_mtm_svd_results, f)


# NOTE: lfv spectra are not normalized to reference data. In order for confidence
#       intervals to be valid, all lfv spectra must be normalized to lfv_ref