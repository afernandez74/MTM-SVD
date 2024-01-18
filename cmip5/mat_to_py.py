#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 09:00:42 2023

@author: afer
"""

# import .mat files with the past1000 CMIP5 data
# used in Mann et al., 2021
import scipy.io
import os 
import pickle
import numpy as np
import xarray as xr

#%% 1) read in the files

# dictionary where data will be stored
cmip5_past1000 = {}

# read files in path
path = '/Volumes/AlejoED/Work/MannSteinman_Proj/Mann_MTMSVD_2020/Data/past1000_cmip5_toPython/'
files = os.listdir(path)
files = [entry for entry in files if not entry.startswith('.')]

# model names
models = [file_name.replace('.mat', '') for file_name in files]

# Load MATLAB .mat files and convert data to numpy dataset dictionary
for model in models:
    mat_contents = scipy.io.loadmat(path+model)

    # Access the structure variable
    mat_structure = mat_contents[model]

    # Convert MATLAB structure to Python dictionary
    dic = {}
    for field_name in mat_structure.dtype.names:
        dic[field_name] = mat_structure[field_name][0, 0]
    
    # save temporary dictionary to final dictionary
    cmip5_past1000[model] = dic
del dic, field_name, model, models, mat_contents, mat_structure
    
#%% 2) transform data into xarray datasets and save as .nc files

for model, data in cmip5_past1000.items():
    # obtain relevant data for coordinates and variables
    lat = data['lat'].flatten()
    lon = data['lon'].flatten()
    year = data['ayear'].flatten()
    tas = data['atas']
    tas = np.transpose(tas,(2,1,0)) # to keep same structure as cmip6 models
    
    # fix datasets with repeat years
    years_unique, counts = np.unique(year,return_counts = True)
    if len(counts) != len(year):
        count_reps = np.count_nonzero(counts == 2)
        rep_yr = years_unique[np.where(counts > 1)]
        if count_reps == 1:
            ix = np.where(year == rep_yr)[0][1]
            year = np.delete(year, ix)
            tas = np.delete(tas,ix,axis=0)
    
        elif count_reps > 1:
            # for c in range(count_reps):
            #     rep_yr_i = rep_yr[c]
            #     ix = np.where(year == rep_yr_i)[0][1]
            #     year = np.delete(year, ix)
            #     tas = np.delete(tas,ix,axis=0)
            year_i = year[0]
            length = tas.shape[0]
            year = np.arange(year_i,year_i+length)
    # create xarray dataarrays
    lon_coord = xr.DataArray(lon, dims='lon', coords={'lon': lon}, attrs={'units': 'degrees_north'})
    lat_coord = xr.DataArray(lat, dims='lat', coords={'lat': lat}, attrs={'units': 'degrees_east'})
    year_coord = xr.DataArray(year, dims='time', coords={'time': year}, attrs={'units': 'year AD'})
    atas_array = xr.DataArray(tas, dims=('time','lat', 'lon'), coords={'lon': lon_coord, 'lat': lat_coord, 'time': year_coord}, attrs={'units': 'deg K'})
    
    # create xarray dataset
    ds_i = xr.Dataset({'tas': atas_array})
    ds_i.attrs['model_name'] = model
    
    # save .nc files
    save_path = os.path.expanduser("~/mtm_local/cmip5_past1000/datasets/")
    file_name = f'{model}_past1000_dataset.nc'
    full_path = save_path + file_name
    ds_i.to_netcdf(full_path)

print (f'Datasets saved to: {save_path}')
