#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:15:34 2024

@author: afer
"""
import pyleoclim as pyleo
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask
from sklearn.linear_model import LinearRegression
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

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean

#%% choose case and initialize parameters
forc = 'unforced' #forced OR unforced
case = "ALL_FORCING"

if forc =='forced':
    
    data = CESM_LME[case].TS
else:
    data = CESM_LME_unforced[case].TS

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(data.lon,data.lat)

freq = 1/10 #filter cutoff freq

#analysis period
period = slice(data.year[0].values+(1/freq), data.year[-1].values-(1/freq))

# coordinates for AMV index

AMV_lat = slice(0,60)
AMV_lon = slice(285,360)

#%% load shapefile and select shape for North Atlantic mask

shpfile = load_shape_file('/Users/afer/mtm_local/misc/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')

NA = select_shape(shpfile, 'NAME', 'North Atlantic Ocean', plot=False)


#%% AMV index of ensemble mean
ds_i = ens_mean.sel(run=0, year = period).TS

ds_mask = ds_i.isel(year = 0).drop('year')

finalMask = xr.full_like(ds_mask, np.nan)

polygon = NA

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

# =============================================================================
# AMV index
# =============================================================================

dat_NA = ds_i.where(NA_mask ==1).sel(lat = AMV_lat, lon = AMV_lon)
dat_NA_ts = dat_NA.mean(dim = ['lat','lon'])

AMV_index_EM = xr.DataArray(
    data = pyleo.Series(
        time = dat_NA_ts.year.values,
        value = dat_NA_ts.values).filter(cutoff_freq=freq).detrend(method = 'savitzky-golay').standardize().value,
    dims = dat_NA_ts.dims,
    coords = dat_NA_ts.coords
)
ds = xr.Dataset()
ds['AMV_index']= AMV_index_EM
ds.to_netcdf(f'~/mtm_local/CESM_LME/AMV_index/EM/{case}.nc')
#%% apply mask and calculate AMV index and regression

# AMV_reg_coef = {}
# AMV_reg_score = {}
AMV_index = {}

for run_i, ds_i in data.groupby('run'):
    
    key = f'{case}_{run_i:03}'
    print (f'masking {key} data')
    
    # =============================================================================
    # apply mask
    # =============================================================================

    ds_mask = ds_i.sel(year = 850).drop('year')
    
    finalMask = xr.full_like(ds_mask, np.nan)
    
    polygon = NA

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

    # =============================================================================
    # AMV index
    # =============================================================================
    
    print (f'calculating AMV index for {key}')
    dat_NA = ds_i.where(NA_mask ==1).sel(lat = AMV_lat, lon = AMV_lon)
    dat_NA_ts = dat_NA.mean(dim = ['lat','lon'])
    
    AMV_index[key] = xr.DataArray(
        data = pyleo.Series(
            time = dat_NA_ts.year.values,
            value = dat_NA_ts.values).filter(cutoff_freq=freq).detrend(method = 'savitzky-golay').standardize().value,
        dims = dat_NA_ts.dims,
        coords = dat_NA_ts.coords
    )

    # =============================================================================
    # linear regression
    # =============================================================================

    # print(f'performing linear regression for {key}')
    # AMV_reg_coef_temp = np.full(ds_i.shape[1:],np.nan)
    # AMV_reg_score_temp = np.full(ds_i.shape[1:],np.nan)

    # for lat_i in ds_i.lat.values:

    #     for lon_i in ds_i.lon.values:

    #         ts = ds_i.sel(lat=lat_i, lon=lon_i)

    #         pyleo_series = pyleo.Series(time=ts.year.values, value=ts.values)
    #         pyleo_series_filtered = pyleo_series.filter(cutoff_freq=freq)  # Filter
    #         pyleo_series_detrended = pyleo_series_filtered.detrend(method = 'savitzky-golay')  # Detrend

    #         x = AMV_index[key].values.reshape(-1,1)
    #         y = pyleo_series_detrended.value
    #         model = LinearRegression().fit(x,y)

    #         AMV_reg_coef_temp[ds_i.lat == lat_i, ds_i.lon == lon_i] = model.coef_[0]
    #         AMV_reg_score_temp[ds_i.lat == lat_i, ds_i.lon == lon_i] = model.score(x, y)
            
    # AMV_reg_coef[f'{case}_{run_i:03}'] = xr.DataArray(
    #     data = AMV_reg_coef_temp,
    #     dims = ds_mask.dims,
    #     coords = ds_mask.coords)

    # AMV_reg_score[key] = xr.DataArray(
    #     data = AMV_reg_score_temp,
    #     dims = ds_mask.dims,
    #     coords = ds_mask.coords)

del finalMask, NA, polygon,shpfile, temp_mask, xx, yy

# save results 
da = xr.concat(AMV_index.values(),dim = 'run').rename('AMV_index')
ds = xr.Dataset()
ds['AMV_index'] = da
ds.to_netcdf(path = f'~/mtm_local/CESM_LME/AMV_index/{forc}/{case}.nc')
#%% maps

fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_score,cmap = 'turbo',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =40,
                  transform = ccrs.PlateCarree())
                  #vmin=-1, vmax = 1)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  label = '')
