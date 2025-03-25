#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:58:57 2025

@author: afer
"""

import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from mtmsvd_funcs import mtm_svd_lfv,mtm_svd_conf,quick_lfv_plot,mtm_svd_bandrecon,reshape_3d_to_2d
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,BoundaryNorm
# Set global settings for plots
mpl.rcParams.update(mpl.rcParamsDefault)
# Set global settings for plots
plt.rcParams.update({
    'font.size': 14,
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    # 'figure.figsize': (8, 4),  # Customize figure size for publication
    # 'figure.dpi': 300,  # Higher DPI for publication quality
    # 'lines.linewidth': 2,  # Line width
    'lines.color': 'black',  # Default line color
    'axes.grid': True,  # Show grid
    'axes.titlesize': 16,  # Axis title font size
    'axes.labelsize': 14,  # Axis label font size
    'xtick.labelsize': 12,  # X-tick label font size
    'ytick.labelsize': 12,  # Y-tick label font size
    'xtick.major.size': 6,  # Major tick size for x-axis
    'ytick.major.size': 6,  # Major tick size for y-axis
    'legend.fontsize': 12,  # Legend font size
    'legend.loc': 'best',  # Best location for the legend
    # 'legend.borderpad': 1.0,  # Padding around the legend
    'grid.linestyle': '--',  # Dashed grid lines
    'grid.color': 'gray',  # Grid line color
    'grid.alpha': 0.5,  # Transparency of grid lines
    'xtick.direction': 'in',  # Tick direction
    'ytick.direction': 'in',  # Tick direction
    'axes.labelpad': 10,  # Padding between axis labels and plot
})

#%% import model data
path = os.path.expanduser('~/mtm_local/cmip6_past1000/datasets/')

# obtain name of file (only works if one file is present only)
files = os.listdir(path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

# standardize time bounds for all analyses (remove first and last years for irregularities)
year_i = 851
year_f = 1849

# create dictionary with all datasets
past1000={}
for file in files:
    # load dataset 
    ds_i = xr.open_dataset(path+file)
    
    # fix name for CESM data due to lack of ".source_id" entry
    if 'source_id' in ds_i.attrs:
        model = ds_i.source_id
    
    past1000[model] = ds_i.sel(year=slice(year_i,year_f)).tas
    
# load North Atlantic mask file 

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)

# %% 2) Compute LFV spectra for all simulations individually

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

NA = False #True if North Atlantic only

EM = False # to analyze ensemble mean only

wndw = 200 # window of MTM-SVD analysis 

rez = 5 # window moves every 'rez' years

wndw2 = np.floor(wndw/2) #half window

year_i = int(year_i + np.floor(wndw/2))
year_f = int(year_f - np.floor(wndw/2))

# =============================================================================
# loop through all models
# =============================================================================

LFV = {}

dat = past1000
    
for model, ds_i in dat.items():

    print('model = ' + model)
    # if NA:
    #     ds_i = ds_i.where(NA_mask == 1)
        
    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')
    
    spectra = []
    yrs = []
    for yr in range (year_i,year_f,rez):
        
        # temp data
        range_yrs = slice(yr - wndw2, yr + wndw2-1)
        print(range_yrs)
        tas = ds_i.sel(year = range_yrs)
        tas_np = tas.to_numpy()
        
        tas_2d = reshape_3d_to_2d(tas_np)
        
        freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
        
        spectrum = xr.DataArray(
            data = lfv,
            dims = ['freq'],
            coords = dict(freq = freq),
            )
        yrs.append(yr)
        spectra.append(spectrum)
    LFV[f'{model}_lfv'] = xr.concat(spectra, dim='mid_yr').assign_coords(mid_yr=yrs)


# merge all dataArrays into a single dataset
lfv = xr.Dataset(LFV)
