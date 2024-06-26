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
file_path = os.path.expanduser("~/mtm_local/past1000/datasets/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

# standardize time bounds for all analyses (remove first and last years for irregularities)
year_i = 851
year_f = 1849

# create dictionary with all datasets
past1000={}
for file in files:
    # load dataset 
    ds_i = xr.open_dataset(file_path+file)
    
    # fix name for CESM data due to lack of ".source_id" entry
    if 'source_id' in ds_i.attrs:
        model = ds_i.source_id
    else:
        model = 'CESM'
    
    # save data to dictionary and rename TS in CESM for comatibility
    # keep only data within desired time bounds
    if model != 'CESM':
        past1000[model] = ds_i.sel(year=slice(year_i,year_f))
    else:
        ds_i = ds_i.rename({'TS': 'tas'})
        past1000[model] = ds_i.sel(year=slice(year_i,year_f))

# get rid of unneeded variables from CESM data 
desired_variable_names = ['lat', 'lon', 'year', 'tas']
past1000['CESM'] = past1000['CESM'][desired_variable_names]

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/past1000/tas_dic/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000, f)
    
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

for key,value in past1000.items():
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

path_results = os.path.expanduser('~/mtm_local/past1000/mtm_svd_results/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CMIP6_past1000_mtm_svd_results{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_mtm_svd_results, f)
    
print(f"results dictionary saved to {path_results}")
    

# %% 3) regrid all data to common resolution

# Show grid sizes for all models 
print("Lat x Lon values for all models:")
for key, value in past1000.items():
    lat = len(past1000[key].lat)
    lon = len(past1000[key].lon)
    print(f"resolution_{key} = {lat}x{lon}")
    
del key, value    

# regrid all entries to CESM grid
past1000_regrid = {}
ref = past1000['CESM']
for key, value in past1000.items():
    ds_i = value
    ds_i_regridded = ds_i.interp_like(ref,method='linear')
    past1000_regrid[key]=ds_i_regridded

print("___")
print("Lat x Lon values for regridded models:")
# Show new grid sizes 
for key, value in past1000_regrid.items():
    lat = len(past1000_regrid[key].lat)
    lon = len(past1000_regrid[key].lon)
    print(f"resolution_{key} = {lat}x{lon}")
del key, value  

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/past1000/tas_dic_regrid/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_regrid, f)
    
print(f"results dictionary saved to {path_results}")


# %% 4) Compute the ensemble mean and unforced data (i.e., forcing series removed)

# concatenate datasets to perform calculations
past1000_concat = xr.concat(list(past1000_regrid.values()), dim='model', coords = 'minimal')

# =============================================================================
# # calculate ensemble mean 
# =============================================================================
past1000_ensemb_mean = past1000_concat.mean(dim='model')

# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/past1000/tas_dic_regrid_ensemb_mean/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'past1000_dic_regrid_ensemb_mean_{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_ensemb_mean, f)

# =============================================================================
# calculate unforced data (forced - ensemble mean)
# =============================================================================
past1000_regrid_unforced = {}
#subtract ensemble mean from each individual dataarray
for key, value in past1000_regrid.items():
    past1000_regrid_unforced[key] = value - past1000_ensemb_mean
    
# save dictionary with data 
path_results = os.path.expanduser('~/mtm_local/past1000/tas_dic_regrid_unforced/')

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
past1000_unforced_mtm_svd_results={}

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
    past1000_unforced_mtm_svd_results[freq_key] =  freq
    past1000_unforced_mtm_svd_results[lfv_key] = lfv
del key, value    
# save results to dic  

path_results = os.path.expanduser('~/mtm_local/past1000/mtm_svd_unforced_results/')

results_timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
results_file_name = f'CMIP6_past1000_mtm_svd_results{results_timestamp}'
results_full_path = path_results+results_file_name

with open(results_full_path, 'wb') as f:
    pkl.dump(past1000_unforced_mtm_svd_results, f)
    
print(f"results dictionary saved to {path_results}")
    

#%% 6) Reconstruct spatial patterns
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

# Select frequency(ies)
# fo = 0.018
fo = 0.2263

# Calculate the reconstruction
model = 'MRI-ESM2-0'
dat = past1000_regrid_unforced['MRI-ESM2-0']

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
fig, (ax1, ax2) = plt.subplots(2,1,subplot_kw={'projection': ccrs.PlateCarree()},figsize=(10,16))
ax1.semilogx(freq, lfv, '-', c='k')
[ax1.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=10)
ax1.set_xlabel('Frequency [1/years]')
# ax1.set_title('LVF at %i m'%d)

ax2.coastlines()
pc = ax2.pcolor(xx, yy, RV, cmap='jet', vmin=0) 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Variance explained')
# ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

plt.tight_layout()
save_name = os.path.expanduser('~/mtm_local/AGU23_figs/MRI_map_nino_unforced')
plt.savefig(save_name, format = 'svg')
plt.title(f'{model} forced model variance explained by period %.0f yrs'%(1./fo))

plt.show()

print('finish')

#%% spectrun
# xticks = [100,60,40,20,10]
xticks = [100,80,60,20,10,7,5,3]

# figure drawing
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,0.9)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
p1 = ax.plot(freq,lfv,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkred',
        label = 'Ensemble mean - Forced')

[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in ci[:,1]]
# save_path = os.path.expanduser(f'~/mtm_local/AGU23_figs/{model}_lfv_')
plt.title(f'{model} spectrum forced ')
# plt.savefig(save_path, format='svg')
