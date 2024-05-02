#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:14:18 2024

@author: afer
"""
import xarray as xr
import os 
import numpy as np
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask

# =============================================================================
# Make North Atlantic mask and save as .nc
# =============================================================================

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



shpfile = load_shape_file('/Users/afer/mtm_local/misc/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')

NA = select_shape(shpfile, 'NAME', 'North Atlantic Ocean', plot=False)
LAB = select_shape(shpfile, 'NAME', 'Labrador Sea', plot=False)
CAR = select_shape(shpfile, 'NAME', 'Caribbean Sea', plot=False)
GUL = select_shape(shpfile, 'NAME', 'Gulf of Mexico', plot=False)

ds = CESM_LME['ALL_FORCING'].TS

ds = ds.squeeze().sel(run = 0, year = 850)

finalMask = xr.full_like(ds, np.nan)

polygon_NA = NA
polygon_LAB = LAB 
polygon_CAR = CAR
polygon_GUL = GUL

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(ds.lon,ds.lat)

print("masking North Atlantic")

temp_mask_NA = serial_mask(xx, yy, polygon_NA)
temp_mask_LAB = serial_mask(xx, yy, polygon_LAB)
temp_mask_CAR = serial_mask(xx, yy, polygon_CAR)
temp_mask_GUL = serial_mask(xx, yy, polygon_GUL)


# set integer for the given region.
temp_mask_NA = xr.DataArray(temp_mask_NA, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking
temp_mask_LAB = xr.DataArray(temp_mask_LAB, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking
temp_mask_CAR = xr.DataArray(temp_mask_CAR, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking
temp_mask_GUL = xr.DataArray(temp_mask_GUL, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking

# Assign NaNs outside of mask, and index within
temp_mask_NA = (temp_mask_NA.where(temp_mask_NA)).fillna(0)
temp_mask_LAB = (temp_mask_LAB.where(temp_mask_LAB)).fillna(0)
temp_mask_CAR = (temp_mask_CAR.where(temp_mask_CAR)).fillna(0)
temp_mask_GUL = (temp_mask_GUL.where(temp_mask_GUL)).fillna(0)

temp_mask = temp_mask_LAB+temp_mask_NA+temp_mask_CAR+temp_mask_GUL
# Add masked region to master array.
finalMask = finalMask.fillna(0) + temp_mask

# Make your zeros NaNs again.
finalMask = finalMask.where(finalMask > 0)

NA_mask = finalMask.sel(lat = slice(0,60), lon = slice(260,360))

NA_mask = xr.DataArray(
    data = NA_mask.values,
    coords = NA_mask.coords,
    dims = NA_mask.dims,
    name = 'NA_mask'
    )

path = os.path.expanduser("~/mtm_local/CESM_LME/masks/")
NA_mask.to_netcdf(path + 'NA_mask.nc')
