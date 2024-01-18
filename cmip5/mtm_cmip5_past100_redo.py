# Script for MTM_SVD analysis of the CMIP6 past1000 experiments
# the script analyzes all 4 members of the ensemble, produces LFV spectra 
# for both forced+internal and internal-only(unforced) series

# %% import functions and packages

from mtm_funcs import *
from readin_funcs_past1000 import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl
import time
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# %% 1) Load annualized datasets and add them to a common dictionary

# file path where dictionary was saved in mtm_preprocessing.py script
file_path = os.path.expanduser("~/mtm_local/cmip5_past1000/datasets_re_download/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

# standardize time bounds for all analyses (remove first and last years for irregularities)
year_i = 851
year_f = 1849

# create dictionary with all datasets
cmip5_past1000={}
for file in files:
    # load dataset 
    ds_i = xr.open_dataset(file_path+file)
    
    model = ds_i.model_id

    #specify physics for GISS members
    if model.startswith('GISS'):
        model = model + '_p' + str(ds_i.physics_version)

    # save data to dictionary and rename TS in CESM for comatibility
    
    cmip5_past1000[model] = ds_i.sel(year=slice(year_i,year_f))


# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/tas_dic/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'cmip5_past1000_dic_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(cmip5_past1000, f)
    
print(f"results dictionary saved to {path_results}")


# %% 2) Compute LFV spectra for all simulations individually

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

# standardize time bounds for all analyses
year_i = 851
year_f = 1849

# =============================================================================
# Loop for MTM-SVD calculations
# =============================================================================

# dictionary to save results
past1000_mtm_svd_results={}

for key,value in cmip5_past1000.items():
    ds_i = value
    model = key
    
    # calculate weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon.values,ds_i.lat.values)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # reshape 'tas' matrix to 2d
    tas = ds_i.sel(year=slice(year_i,year_f))
    tas_3d = tas.tas.to_numpy()
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
    past1000_mtm_svd_results[freq_key] =  freq
    past1000_mtm_svd_results[lfv_key] = lfv
del key, value    
# save results to dic  

path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/mtm_svd_results/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CMIP5_past1000_mtm_svd_results{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_mtm_svd_results, f)
    
print(f"results dictionary saved to {path_results}")
    

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
cmip5_past1000_regrid = {}
ref = cmip5_past1000['GISS-E2-R_p121']
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
path = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'cmip5_past1000_dic_regrid_{results_timestamp}'
results_full_path = path+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(cmip5_past1000_regrid, f)
    
print(f"results dictionary saved to {path}")

del ds_i, ds_i_regridded, f, lat, lon, path, results_file_name, results_timestamp

# %% 4) Compute the ensemble mean and unforced data (i.e., forcing series removed)

# concatenate datasets to perform calculations
# only use one GISS iteration
cmip5_past1000_regrid_oneGISS = cmip5_past1000_regrid
keys_drop = [ 'GISS-E2-R_p1221',
 'GISS-E2-R_p122',
 'GISS-E2-R_p123',
 'GISS-E2-R_p124',
 'GISS-E2-R_p121',
 'GISS-E2-R_p126',
 'GISS-E2-R_p127',
 'GISS-E2-R_p128']
keys_to_drop = [key for key in cmip5_past1000_regrid.keys() if any(string in key for string in keys_drop)]
    
for key in keys_to_drop:
    del cmip5_past1000_regrid_oneGISS[key]

# choose what ensemble mean to use 
# past1000_concat = xr.concat(list(cmip5_past1000_regrid.values()), dim='model', coords = 'minimal',compat='override')
past1000_concat_oneGISS = xr.concat(list(cmip5_past1000_regrid_oneGISS.values()), dim='model', coords = 'minimal',compat='override')

# =============================================================================
# # calculate ensemble mean 
# =============================================================================
past1000_ensemb_mean = past1000_concat_oneGISS.mean(dim='model')

# save dictionary with data 
path = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid_ensemb_mean_oneGISS/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_ensemb_mean_{results_timestamp}'
results_full_path = path+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_ensemb_mean, f)
    
del path, results_timestamp, results_file_name, results_full_path
# =============================================================================
# calculate unforced data (forced - ensemble mean)
# =============================================================================
past1000_regrid_unforced = {}
#subtract ensemble mean from each individual dataarray
for key, value in cmip5_past1000_regrid.items():
    past1000_regrid_unforced[key] = value - past1000_ensemb_mean
    
# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/tas_dic_regrid_unforced_oneGISS/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_unforced_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_regrid_unforced, f)


# %% 5) Compute the LVF spectra for 'unforced' datasets

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

# standardize time bounds for all analyses
year_i = 851
year_f = 1849

# =============================================================================
# Loop for MTM-SVD calculations
# =============================================================================

# dictionary to save results
cmip5_past1000_unforced_mtm_svd_results_oneGISS={}

for key, value in past1000_regrid_unforced.items():
    ds_i = value
    model = key
    
    # calculate weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon.values,ds_i.lat.values)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # reshape 'tas' matrix to 2d
    tas = ds_i.sel(year=slice(year_i,year_f))
    tas_3d = tas.tas.to_numpy()
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
    cmip5_past1000_unforced_mtm_svd_results_oneGISS[freq_key] =  freq
    cmip5_past1000_unforced_mtm_svd_results_oneGISS[lfv_key] = lfv
del key, value    
# save results to dic  

path_results = os.path.expanduser('~/mtm_local/cmip5_past1000/new_analysis/mtm_svd_unforced_results_oneGISS/')

results_file_name = 'CMIP6_past1000_mtm_svd_results'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(cmip5_past1000_unforced_mtm_svd_results_oneGISS, f)
    
print(f"results dictionary saved to {path_results}")
    


#%% 6) Reconstruct spatial patterns
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 


# Select frequency(ies)
fo = 0.018

# Calculate the reconstruction
model = 'IPSL-CM5A-LR'
dat =cmip5_past1000[model]

# grid
[xx,yy] = np.meshgrid(dat.lon.values,dat.lat.values)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# reshape 'tas' matrix to 2d
tas = dat.sel(year=slice(year_i,year_f))
tas_3d = tas.tas.to_numpy()
tas_2d = reshape_3d_to_2d(tas_3d)

freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)

print(f'total variance explained by {fo} ={totvarexp}')

# Plot the map for each frequency peak

RV = np.reshape(vexp,xx.shape, order='F')
fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[1,3]},subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,16))
ax1.semilogx(freq, lfv, '-', c='k')
# [ax1.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in adj_ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=10)
ax1.set_xlabel('Frequency [1/years]')
# ax1.set_title('LVF at %i m'%d)

ax2.coastlines()
pc = ax2.pcolor(xx, yy, RV, cmap='jet', vmin=0) 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Variance explained')
# ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

plt.tight_layout()
save_name = os.path.expanduser('~/mtm_local/AGU23_figs/map_CESM_nino_obs')
# plt.savefig(save_name, format = 'svg')
plt.title('ACCESS model variance explained by period %.0f yrs'%(1./fo))

plt.show()
# plt.clf()


print('finish')


