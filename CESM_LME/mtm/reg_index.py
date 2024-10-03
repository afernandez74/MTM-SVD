#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 09:10:31 2024

@author: afer
"""
import pyleoclim as pyleo
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask
from sklearn.linear_model import LinearRegression
import xskillscore as xss
import os
from mtmsvd_funcs import butter_lowpass

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
# path to the climate dataset to be utilized
path = "/Users/afer/mtm_local/misc/hadcrut5/HadCRUT5_anom_ens_mean.nc"
# files = listdir(path)   
# files.sort()

print('Load in data from NetCDF files...')

ds_monthly = xr.open_dataset(path)['tas_mean']

# calculate annual means
ds = ds_monthly.groupby('time.year').mean(dim = 'time')

# obtain lat and lon
lon = ds.longitude
lat = ds.latitude

# Plot map of the variable
model = 'had5_obs_temp'
xgrid, ygrid = np.meshgrid(lon,lat)

RV = ds.mean(dim=['year'])
fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='grey')

p = RV.plot(cmap = 'turbo',
                 add_colorbar = False,
                 transform = ccrs.PlateCarree())

cb = plt.colorbar(p, orientation='horizontal', pad=0.05,shrink=0.8,label = 'Mean temp change')
plt.title('map of observed warming')
p.axes.coastlines()

# timesries
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ds.mean(dim = ['latitude','longitude']).plot()

#%% load North Atlantic mask 

shpfile = load_shape_file('/Users/afer/mtm_local/misc/GOaS_v1_20211214/goas_v01.shp')

NA = select_shape(shpfile, 'name', 'North Atlantic Ocean', plot=False)
OCEANS = [select_shape(shpfile, 'name', name, plot=False)for name in shpfile.name.values]

ds_mask = ds.sel(year = 1850)

finalMask = xr.full_like(ds_mask, np.nan)

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(ds.longitude,ds.latitude)

print("masking North Atlantic")

temp_mask = serial_mask(xx, yy, NA)
temp_masks_OCEANS = [serial_mask(xx, yy, poly) for poly in OCEANS]

# set integer for the given region.
temp_mask = xr.DataArray(temp_mask, dims=['latitude', 'longitude']) # dims should be like your base data array you'll be masking

# Assign NaNs outside of mask, and index within
temp_mask = (temp_mask.where(temp_mask)).fillna(0)

# Add masked region to master array.
finalMask = finalMask.fillna(0) + temp_mask

# Make your zeros NaNs again.
finalMask = finalMask.where(finalMask > 0)

NA_mask = finalMask

del finalMask, NA, shpfile, temp_mask, xx, yy

#%% Gloabal SST anomaly (60S - 60N)

freq = 1/10 #filter frequency for all data
corr_period = slice(1850,2020) #period of analysis for all

dat_glob = ds.sel(latitude=slice(-60,60))
dat_glob_ts = dat_glob.mean(dim = ['latitude','longitude'])

weights = np.cos(np.deg2rad(dat_glob.latitude))
weights.name = "weights"

glob_anom = xr.DataArray(
    data = pyleo.Series(
        time = dat_glob_ts.year.values,
        value = dat_glob_ts.values).filter(cutoff_freq=freq).value,
    dims = dat_glob_ts.dims,
    coords = dat_glob_ts.coords
).sel(year = corr_period)

dat_glob_w = dat_glob.weighted(weights)
dat_glob_w_ts = dat_glob_w.mean(dim = ['latitude','longitude'])

glob_anom_w = xr.DataArray(
    data = pyleo.Series(
        time = dat_glob_w_ts.year.values,
        value = dat_glob_w_ts.values).filter(cutoff_freq=freq).detrend(
            # method = 'savitzky-golay'
            ).value,
    dims = dat_glob_w_ts.dims,
    coords = dat_glob_w_ts.coords
).sel(year = corr_period)

#%% AMV index

dat_NA = ds.where(NA_mask ==1).sel(
    latitude = slice(0,60), 
    longitude = slice(-75,0), 
    year = corr_period).tas_mean

weights = np.cos(np.deg2rad(dat_NA.latitude))
weights.name = "weights"

dat_NA_w = dat_NA.weighted(weights)

dat_NA_ts = dat_NA_w.mean(dim = ['latitude','longitude'])

NA_resid = dat_NA_ts - glob_anom_w

AMV_index = xr.DataArray(
    data = butter_lowpass(NA_resid, cutoff_frequency = freq, sampling_frequency = 1),
    dims = NA_resid.dims,
    coords = NA_resid.coords)
    

AMV_index = xr.DataArray(
    data = pyleo.Series(
        time = dat_NA_ts.year.values,
        value = dat_NA_ts.values).filter(cutoff_freq=freq).value,
    dims = dat_NA_ts.dims,
    coords = dat_NA_ts.coords
).sel(year = corr_period)

dat_NA_w = dat_NA.weighted(weights)
dat_NA_w_ts = dat_NA_w.mean(dim = ['latitude','longitude'])

AMV_index_w = xr.DataArray(
    data = pyleo.Series(
        time = dat_NA_w_ts.year.values,
        value = dat_NA_w_ts.values).filter(cutoff_freq=freq).value,
    dims = dat_NA_ts.dims,
    coords = dat_NA_ts.coords
).sel(year = corr_period)

#%% process the dataset so only timeseries that don't contain more than 10% NaN values
# are kept, and then the missing data is linearly interpolated
mask = ds.notnull().mean(dim='year') >= 0.5

ds_good = ds.where(
    mask, drop=True).interpolate_na(
        dim='year', method='linear')
        # .bfill(
        #     dim = 'year').ffill(dim  = 'year')
            
# data leftover are detrended and filtered for analysis
processed_data = np.full(ds_good.shape, np.nan, dtype=np.float64)

for lat in ds_good.latitude.values:

    for lon in ds_good.longitude.values:

        ts = ds_good.sel(latitude=lat, longitude=lon)
        
        if np.isnan(ts.values).all():
            processed_data[:, ds_good.latitude == lat, ds_good.longitude == lon] = ts.values.reshape((ts.values.shape[0],1))
        
        elif np.isnan(ts.values).any():

            pyleo_series = pyleo.Series(
                time=ts.year.where(~np.isnan(ts)).values, 
                value=ts.where(~np.isnan(ts)).values)
            pyleo_series_filtered = pyleo_series.filter(cutoff_freq=freq)  # Filter
            pyleo_series_detrended = pyleo_series_filtered.detrend()
                # method = 'savitzky-golay')  # Detrend

            processed_data[~np.isnan(ts).values, ds_good.latitude == lat, ds_good.longitude == lon] = pyleo_series_detrended.value

        else:

            pyleo_series = pyleo.Series(time=ts.year.values, value=ts.values)
            pyleo_series_filtered = pyleo_series.filter(cutoff_freq=freq)  # Filter
            pyleo_series_detrended = pyleo_series_filtered.detrend()
                # method = 'savitzky-golay')  # Detrend

            processed_data[:, ds_good.latitude == lat, ds_good.longitude == lon] = pyleo_series_detrended.standardize().value.reshape((pyleo_series_filtered.value.shape[0],1))
            

ds_proc = xr.DataArray(
    data=processed_data,
    dims=['year', 'latitude', 'longitude'],
    coords={
        'year': ds_good.year,
        'latitude': ds_good.latitude,
        'longitude': ds_good.longitude
    }
).sel(year = corr_period)

del processed_data,ds_good, pyleo_series, pyleo_series_detrended, pyleo_series_filtered,ts

#%% calculate correlation between AMV index and processed global temperature data

# get rid of last few years of warming

# correlation calculation
AMV_corr = xr.corr(ds_proc,AMV_index,dim = 'year')

AMV_corr_p_val = xss.pearson_r_p_value(ds_proc,
                                       AMV_index,
                                       dim = 'year',
                                       skipna = False,
                                       keep_attrs = True)

AMV_corr_eff_p_val = xss.pearson_r_eff_p_value(ds_proc,
                                       AMV_index,
                                       dim = 'year',
                                       skipna = False,
                                       keep_attrs = True)

#%% calculate linear regression 

AMV_reg_coef = np.full(AMV_corr.shape, np.nan, dtype=np.float64)
AMV_reg_score = np.full(AMV_corr.shape, np.nan, dtype=np.float64)

for lat in ds_proc.latitude:
    for lon in ds_proc.longitude:

        ts = ds_proc.sel(latitude=lat, longitude=lon)

        if np.isnan(ts.values).all():
            AMV_reg_coef[ds_proc.latitude == lat, ds_proc.longitude == lon] = np.nan
            AMV_reg_score[ds_proc.latitude == lat, ds_proc.longitude == lon] = np.nan

        elif np.isnan(ts.values).any():

            x = AMV_index.where(~np.isnan(ts.values)).values
            x = x[~np.isnan(x)].reshape(-1,1)
            y = ts.values
            y = y[~np.isnan(y)]
            model = LinearRegression().fit(x,y)
            
            
        else:
            x = AMV_index.values.reshape(-1,1)
            y = ts.values
            model = LinearRegression().fit(x,y)

        AMV_reg_coef[ds_proc.latitude == lat, ds_proc.longitude == lon] = model.coef_[0]
        AMV_reg_score[ds_proc.latitude == lat, ds_proc.longitude == lon] = model.score(x, y)

AMV_reg_coef = xr.DataArray(
    data = AMV_reg_coef,
    dims = AMV_corr.dims,
    coords = AMV_corr.coords)

AMV_reg_score = xr.DataArray(
    data = AMV_reg_score,
    dims = AMV_corr.dims,
    coords = AMV_corr.coords)

del x,y,model,ts
#%%plot correlation map

sig = 0.2#significance level

fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
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
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')


# # # put dots where p_value is below significance level 
significant_lats, significant_lons = np.where(AMV_corr_p_val < sig) 
significant_lats2, significant_lons2 = np.where(ds.notnull().mean(dim='year') >= 0.95)

for lat, lon in zip(significant_lats, significant_lons):
    actual_lat = AMV_corr_p_val.latitude.values[lat]
    actual_lon = AMV_corr_p_val.longitude.values[lon]
    # plt.plot(actual_lon, actual_lat, 'o', color='black', markersize=2, transform=ccrs.Geodetic())

for lat, lon in zip(significant_lats2, significant_lons2):
    actual_lat = ds.latitude.values[lat]
    actual_lon = ds.longitude.values[lon]
    plt.plot(actual_lon, actual_lat, '*', color='gray', markersize=2, transform=ccrs.Geodetic())
    
plt.plot(-50,70,marker = '*',markersize = 10,transform = ccrs.Geodetic())



plt.title('AMV lin reg score HadCRUT5 tas')


save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/corr/')
name = 'AMV_index_HadCRUT5_score'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    
#%% plot AMV index

fig = plt.figure(figsize=(15,15))

ax = plt.subplot(211)
ax.grid(True,which='major',axis='both')
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

p1 = AMV_index.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'black',
        label = 'AMV index')

pos_ix = np.where(AMV_index.values >= 0)
neg_ix = np.where(AMV_index.values <= 0)

f1 = ax.fill_between(AMV_index.year.values, 
                     AMV_index.values,
                     0.0,
                     color = 'red',
                     where= AMV_index.values >0,
                     interpolate = False)

f2 = ax.fill_between(AMV_index.year.values, 
                     0.0,
                     AMV_index.values,
                     color = 'blue',
                     where = AMV_index.values < 0,
                     interpolate = False)


ax.set_title('AMV index HadCRUT5 tas')
ax.set_xlabel('Year CE')
ax.set_ylabel('Index')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/AMV_index/')
name = 'AMV_index HadCRUT5 tas'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
