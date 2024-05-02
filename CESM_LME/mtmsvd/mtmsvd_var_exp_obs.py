#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:36:14 2024

@author: afer
"""

# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

#%% Load datasets 

#%% Load datasets 
# path to the climate dataset to be utilized
path = "/Users/afer/mtm_local/misc/hadcrut5/HadCRUT5_anom_ens_mean.nc"
print('Load in data from NetCDF files...')

ds_monthly = xr.open_dataset(path)['tas_mean']
dt = 1 # yearly data (monthly values averaged below)

# calculate annual means
ds = ds_monthly.groupby('time.year').mean(dim = 'time')

# obtain lat and lon
lon = ds.longitude
lat = ds.latitude

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
#%%#%% recons one simulation only
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

fo = 1/51

# Calculate the reconstruction
# Weights based on latitude
[xx,yy] = np.meshgrid(lon,lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# temp data
tas_np = ds.to_numpy()
tas_2d = reshape_3d_to_2d(tas_np)

freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)

spectrum = xr.DataArray(lfv, coords = [freq], dims=['freq'])

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)

RV = np.reshape(vexp,xx.shape, order='F')

RV = xr.DataArray(
    data = RV,
    dims = ('latitude','longitude'),
    coords=dict(
        longitude=ds.longitude.values,
        latitude=ds.latitude.values),
    attrs=dict(
        description=f"Variance explained by {1./fo:.2f}",
        units="%"), 
    name = f'Var exp obs {1./fo:.2f} yr period'
)

print(f'total variance explained by {fo} = {totvarexp}')

# PLOTS

# =============================================================================
# spectrum
# =============================================================================
fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot(211)
xticks = [100,80,60,40,30,10,5,3]

# figure drawing

ax1.set_xscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax1.set_xticklabels(xticks2_labels)
ax1.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
# plt.ylim(0.4,0.8)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.minorticks_off()

p1 = spectrum.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue')
ci = np.load(os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/conf_int_HadCRUT5.npy'))

[ax1.axhline(y=i, color='black', linestyle='--', alpha=.8, zorder = 1) for i in ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=20, zorder = 20)
ax1.legend()
ax1.set_title('LFV spectrum obs')

ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map 
# =============================================================================

ax2 = plt.subplot(212,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')

p = RV.plot.contourf(ax = ax2,
            add_colorbar = False,
            transform = ccrs.PlateCarree(),
            # vmin = 0, vmax = 2,
            robust = True,
            levels = 40,
            cmap = 'jet')

ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)

# add separate colorbar
cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')

gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')


p.axes.coastlines(color='black')

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)

save_path = os.path.expanduser('~/mtm_local/SRS/figs/')
name = f'recons_{1./fo:.1f}yr'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

