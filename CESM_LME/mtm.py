# calculate the LFV spectra of all members of the CESM LME ensemble

# %% import functions and packages

from mtm_funcs import *
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
    
del case, ds, file, path, path_case

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
for case_i, ds_i in CESM_LME.items():
    print('case='+case_i)
    
    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # temp data
    tas = ds_i.sel(year=slice(year_i,year_f))
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
lfv.to_netcdf(path+'lfv.nc')

# %% Confidence intervals (CNTL)

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# CNTL data
tas_ref = CESM_LME['ALL_FORCING'].isel(run = 0).sel(year=slice(year_i,year_f))
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
np.save(path+'conf_int_CNTL.npy',conflevels)

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

