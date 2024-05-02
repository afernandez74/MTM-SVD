#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:31:59 2024

@author: afer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:50:09 2024

@author: afer
"""

# Calculate the LFV spectra of all members of the CESM LME ensemble once the
# ensemble means have been removed (for ALL_FORCING and VOLCANIC ensembles only)


# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask

#%% Load datasets

path = os.path.expanduser("~/mtm_local/CESM_LME/data/")

cases = [entry for entry in os.listdir(path) if not entry.startswith('.')]

CESM_LME = {}

for case in cases:
    path_case = path+case+'/'
    print(path_case)
    file = [entry for entry in os.listdir(path_case) if not entry.startswith('.')][0]
    ds = xr.open_mfdataset(path_case+file)
    CESM_LME[case] = ds
    
del case, ds, file, path, path_case

#%% Subtract ensemble means from each simulation 

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean

#%% load North Atlantic mask 

shpfile = load_shape_file('/Users/afer/mtm_local/misc/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')

NA = select_shape(shpfile, 'NAME', 'North Atlantic Ocean', plot=False)
LAB = select_shape(shpfile, 'NAME', 'Labrador Sea', plot=False)

ds = CESM_LME['ALL_FORCING']

ds = ds.squeeze().sel(run = 0, year = 850)

finalMask = xr.full_like(ds.TS, np.nan)

polygon_NA = NA
polygon_LAB = LAB 

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(ds.lon,ds.lat)

print("masking North Atlantic")

temp_mask_NA = serial_mask(xx, yy, polygon_NA)
temp_mask_LAB = serial_mask(xx, yy, polygon_LAB)

# set integer for the given region.
temp_mask_NA = xr.DataArray(temp_mask_NA, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking
temp_mask_LAB = xr.DataArray(temp_mask_LAB, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking

# Assign NaNs outside of mask, and index within
temp_mask_NA = (temp_mask_NA.where(temp_mask_NA)).fillna(0)
temp_mask_LAB = (temp_mask_LAB.where(temp_mask_LAB)).fillna(0)

temp_mask = temp_mask_LAB+temp_mask_NA
# Add masked region to master array.
finalMask = finalMask.fillna(0) + temp_mask

# Make your zeros NaNs again.
finalMask = finalMask.where(finalMask > 0)

NA_mask = finalMask
del finalMask,ds, NA, polygon_NA, polygon_LAB,shpfile, temp_mask, xx, yy

#%% MTM_SVD analysis 

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

year_i = 851
year_f = 1849

# =============================================================================
# loop through all cases
# =============================================================================
LFV = {}
for case_i, ds_i in CESM_LME_unforced.items():
    print('case='+case_i)
    
    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # temp data
    tas = ds_i.sel(year=slice(year_i,year_f)).where(NA_mask == 1)
    tas_np = tas.TS.to_numpy()
    
    #cntl
    if tas_np.ndim == 3:

        tas_2d = reshape_3d_to_2d(tas_np)
        print(f'calculating LFV spectrum {case_i}')
        freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)

        spectrum = xr.DataArray(
            data = lfv,
            dims = 'freq',
            coords = [freq],
            name = f'{case_i}_lfv'
            )

        LFV[f'{case_i}_lfv'] = spectrum


    #other cases
    else:

        #loop through "run" dimension, which is assumed to be the shortest one
        for i in range (tas_np.shape[np.argmin(tas_np.shape)]):

            tas_2d = reshape_3d_to_2d(tas_np[:,i,:,:])
            print(f'calculating LFV spectrum {case_i} no. {i+1}')
            freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
            spectrum = xr.DataArray(
                data = lfv,
                dims = 'freq',
                coords = [freq],
                name = f'{case_i}_{(i+1):03d}_lfv'
                )

            LFV[f'{case_i}_00{i+1}_lfv'] = spectrum

# merge all dataArrays into a single dataset
lfv = xr.Dataset(LFV)

#save results dataset as .nc file
path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv/")
lfv.to_netcdf(path+'lfv_unforced_NA.nc')

# %% Confidence intervals (CNTL)

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# CNTL data
tas_ref = CESM_LME_unforced['ALL_FORCING'].isel(run = 0).sel(year=slice(year_i,year_f)).where(NA_mask == 1)
tas_ref_np = tas_ref.TS.to_numpy()

tas_ref_2d = reshape_3d_to_2d(tas_ref_np)

# Weights based on latitude
[xx,yy] = np.meshgrid(tas_ref.lon,tas_ref.lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# conflevels -> 1st column secular, 2nd column non secular 
print(f'Confidence Interval Calculation for CNTL case ({niter} iterations)...')
[conffreq, conflevels] = mtm_svd_conf(tas_ref_2d,nw,kk,dt,niter,sl,w)

#save results
np.save(path+'conf_int_NA_unforced.npy',conflevels)
