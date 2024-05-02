#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:37:19 2024

@author: afer
"""

import numpy as np
import xarray as xr
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask
#%% plotting parameters
# modify global setting
SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE = 50

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

del BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE


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
#%% load SST data
# define function to get you the data you want relatively quickly 

def read_dat(files, variables, pop=False):
    def preprocess(ds):
        return ds[variables].reset_coords(drop=True) # reset coords means they are reset as variables

    ds = xr.open_mfdataset(files, parallel=True, preprocess=preprocess,
                           chunks={'time':1, 'nlon': -1, 'nlat':-1},
                           combine='by_coords')
    if pop==True:
        file0 = xr.open_dataset(files[0])
        ds.update(file0[['ULONG', 'ULAT', 'TLONG', 'TLAT']])
        file0.close()

    ds
    return ds

# define function to be able to plot POP output properly on cartopy projections
def pop_add_cyclic(ds):
    
    nj = ds.TLAT.shape[0]
    ni = ds.TLONG.shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds.TLONG.data
    tlat = ds.TLAT.data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon)    
    lon  = np.concatenate((tlon, tlon + 360.), 1)
    lon = lon[:, xL:xR]

    if ni == 320:
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.))
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    TLAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    TLONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({'TLAT': TLAT, 'TLONG': TLONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
            
    return dso

#%%
path = os.path.expanduser('~/mtm_local/CESM_LME/data/SST/')
files = [path+file for file in os.listdir(path)]
dat = read_dat(files,'SST',pop=True)
dat2= pop_add_cyclic(dat)
SST = dat2.SST
TLAT = dat2.TLAT
TLONG = dat2.TLONG


#%% load North Atlantic mask tas

shpfile = load_shape_file('/Users/afer/mtm_local/misc/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')

NA = select_shape(shpfile, 'NAME', 'North Atlantic Ocean', plot=False)

ds = CESM_LME['ALL_FORCING'].TS

ds = ds.squeeze().sel(run = 0, year = 850)

finalMask = xr.full_like(ds, np.nan)

polygon = NA

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(ds.lon,ds.lat)

print("masking North Atlantic")

temp_mask = serial_mask(xx, yy, polygon)

# set integer for the given region.
temp_mask = xr.DataArray(temp_mask, dims=['lat', 'lon']) # dims should be like your base data array you'll be masking

# Assign NaNs outside of mask, and index within
temp_mask = (temp_mask.where(temp_mask)).fillna(0)

# Add masked region to master array.
finalMask = finalMask.fillna(0) + temp_mask

# Make your zeros NaNs again.
finalMask = finalMask.where(finalMask > 0)

NA_mask = finalMask

del finalMask,ds, polygon, temp_mask, xx, yy, shpfile, NA


#%% compare data

TAS_NA = CESM_LME['ALL_FORCING'].TS.isel(run = 0).where(NA_mask == 1)
SST_NA = SST.where(TLAT>=0).where(TLAT<60).where(TLONG<=0).where(TLONG>=-70)
SST_NA = SST_NA.groupby('time.year').mean('time')

TAS_NA_ts = TAS_NA.mean(dim = ['lat','lon'])
SST_NA_ts = SST_NA.mean(dim = ['nlat','nlon'])
#%%
fig = plt.figure(figsize = (15,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

plt.scatter(TAS_NA_ts.values,SST_NA_ts.values.squeeze()+273.15,
            linewidth = 2,
            color = 'black')
