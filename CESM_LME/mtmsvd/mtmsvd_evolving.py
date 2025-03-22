#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 08:52:28 2024

@author: afer
"""

# calculate the LFV spectra of all members of the CESM LME ensemble

# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)

del case, ds, file, path, path_case

#%% Subtract ensemble means from each simulation 

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean
        
#%% Ensemble mean dictionary
CESM_LME_EM = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        CESM_LME_EM[case]  = case_ds.mean(dim = 'run')


#%% Evolving LFV spectra 

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

unforced = False # True if residual datasets with ensemble mean removed

NA = False #True if North Atlantic only

EM = False # to analyze ensemble mean only

wndw = 200 # window of MTM-SVD analysis 

rez = 5 # window moves every 'rez' years

wndw2 = np.floor(wndw/2) #half window

ds_ref = CESM_LME['CNTL'].TS

year_i = int(ds_ref.year[0].values + np.floor(wndw/2))
year_f = int(ds_ref.year[-1].values - np.floor(wndw/2))

# =============================================================================
# loop through all cases
# =============================================================================

LFV = {}

if unforced:
    dat = CESM_LMEd_unforced
elif EM:
    dat = CESM_LME_EM
else:
    dat = CESM_LME
    
for case_i, ds_i in dat.items():

    print('case='+case_i)
    ds_i = ds_i.TS
    if NA:
        ds_i = ds_i.where(NA_mask == 1)
    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')
    
    #CONTROL
    if case_i == 'CNTL' or EM:
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
        LFV[f'{case_i}_lfv'] = xr.concat(spectra, dim='mid_yr').assign_coords(mid_yr=yrs)


    else:

        #loop through "run" dimension, which is assumed to be the shortest one
        for run_i in range(len(ds_i.run)):
            print(f'{case_i} run: {run_i:03}')
            yrs = []
            spectra = []
            for yr in range (year_i,year_f,rez):
                
                range_yrs = slice(yr - wndw2, yr + wndw2-1)
                print(range_yrs)
                tas = ds_i.sel(year = range_yrs, run = run_i)
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
    
            LFV[f'{case_i}_{run_i:03}_lfv'] = xr.concat(spectra, dim='mid_yr').assign_coords(mid_yr=yrs)

# merge all dataArrays into a single dataset
lfv = xr.Dataset(LFV)

#save results dataset as .nc file
path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv_evo/")
name = f'lfv_evo_{rez}yr_{wndw}window'

name  = name + '_EM' if EM else name
if not EM:
    if unforced:
        name= name + '_unforced'
    else:
        name = name + '_forc'

if NA:
    name = name + '_NA'

lfv.to_netcdf(path + name + '.nc')

# %% Confidence intervals 

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels
wndw = 200 # window of MTM-SVD analysis 

case = 'ALL_FORCING'
run = 6
unforced = False
NA = False
ctr_yr = 1400 #center year for confidence interval calculation


wndw2 = int(np.floor(wndw/2)) #half window
range_yrs = slice(ctr_yr - wndw2, ctr_yr + wndw2-1)

# ref sim
dat = CESM_LME_unforced if unforced else \
    CESM_LME

tas_ref = dat[case].isel(run = run).sel(year=slice(ctr_yr-wndw2,ctr_yr+wndw2)) if not NA else \
    dat[case].isel(run = run).where(NA_mask == 1).sel(year=slice(ctr_yr-wndw2,ctr_yr+wndw2))

tas_ref_np = tas_ref.TS.to_numpy()

tas_ref_2d = reshape_3d_to_2d(tas_ref_np)

# Weights based on latitude
[xx,yy] = np.meshgrid(tas_ref.lon,tas_ref.lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# conflevels -> 1st column secular, 2nd column non secular 
print(f'Confidence Interval Calculation for ref sim ({niter} iterations)...')
[conffreq, conflevels] = mtm_svd_conf(tas_ref_2d,nw,kk,dt,niter,sl,w)

#save results
path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv_evo/")

name = f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_unforced_NA.npy' if unforced and NA else \
        f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_unforced.npy' if unforced and not NA else \
        f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_NA.npy' if NA and not unforced else \
        f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}.npy' 
        
np.save(path+name ,conflevels)

# # =============================================================================
# # Rescaling of confidence intervals 
# # =============================================================================

# fr_sec = nw/(tas_ref_2d.shape[0]*dt) # secular frequency value
# fr_sec_ix = np.where(conffreq < fr_sec)[0][-1] 

# # load CNTL lfv 
# path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv/")
# lfv_ref = xr.open_dataset(path + 'CNTL.nc')

# #calculate mean for non-secular band only 
# lfv_mean = lfv_ref.isel(freq=slice(fr_sec_ix,-1)).mean()
# lfv_mean = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 
# mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

# adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
# adj_ci = conflevels * adj_factor # adjustment for confidence interval values

# np.save(path+'conf_int_CNTL.npy',conflevels)

