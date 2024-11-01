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
    
del case, ds, file, path, path_case
# load North Atlantic mask file 

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)

#%% Subtract ensemble means from each simulation 

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])
        CESM_LME_unforced[case] = case_ds - ens_mean

#%% MTM_SVD analysis 

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

NA = True # Analyze North Atlantic surface temperature only 

unforced = True # Analyze the unforced data

# If chunk == True, input date range for analysis in years

chunk = True # Analyze a specific timperiod of the data

#if chunk, define the period of analysis
year_i = 1150
year_f = 1400

LFV = {}

dat = CESM_LME_unforced if unforced else CESM_LME

#loop through each case of CESM
for case_i, ds_i in dat.items():
    
    ds_i = ds_i.TS
    # obtain NA masked data if NA
    ds_i = ds_i.where(NA_mask==1) if NA else ds_i

    print('case='+case_i)

    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')



    # temp data
    if chunk:
        tas = ds_i.sel(year=slice(year_i,year_f))
    else:
        tas = ds_i.isel(year=slice(1,-1))
    tas_np = tas.to_numpy()

    # control sim:
    if not unforced and tas_np.ndim == 3:

        tas_2d = reshape_3d_to_2d(tas_np)
        print(f'calculating LFV spectrum {case_i}')

        # calculate the LFV spectrum
        freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
        
        # assign results to xr dataarray
        spectrum = xr.DataArray(
            data = lfv,
            dims = 'freq',
            coords = [freq],
            name = f'{case_i}_lfv'
            )

        LFV[f'{case_i}_lfv'] = spectrum

    # other cases
    else:

        #loop through "run" dimension, which is assumed to be the shortest one
        for i in range (tas_np.shape[np.argmin(tas_np.shape)]):

            tas_2d = reshape_3d_to_2d(tas_np[:,i,:,:])
            print(f'calculating LFV spectrum {case_i} no. {i+1}')

            # calculate the LFV spectrum
            freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
            
            # assign results to xr dataarray
            spectrum = xr.DataArray(
                data = lfv,
                dims = 'freq',
                coords = [freq],
                name = f'{case_i}_{(i+1):03d}_lfv'
                )

            LFV[f'{case_i}_{i:03}_lfv'] = spectrum

# merge all dataArrays into a single dataset|
lfv = xr.Dataset(LFV)
lfv.attrs["period"] = f'{year_i}-{year_f}'

# name file to save
name = 'lfv'
name = name + f'_{year_i:04}-{year_f:04}'if chunk else name
name = name + '_NA' if NA else name
name = name + '_unforc' if unforced else name

#save results dataset as .nc file
path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')
lfv.to_netcdf(path + name +'.nc')

# =============================================================================
# quick plot of spectra
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xscale('log')
for var_name, da in lfv.data_vars.items():
    da.plot(xlim = [0.01,0.5],ax =ax)
ax.set_title(f'AF_ONLY_{year_i}-{year_f}\ {name}')


# %% Confidence intervals 

# =============================================================================
# Values for Monte Carlo analysis
# =============================================================================

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

NA = False
chunk = True
unforced = False

#chunk years
year_i = 1600
year_f = 1849

case = 'ALL_FORCING'
run = 6

# ref sim
dat = CESM_LME_unforced if unforced else CESM_LME
dat = dat[case]
ds = dat.sel(lat = slice(-60,60))
ds = ds.where(NA_mask == 1) if NA else ds

if chunk:
    tas_ref = ds.sel(run = run).sel(year=slice(year_i,year_f))
else:
    tas_ref = ds.sel(run = run).isel(year=slice(1,-1))
tas_ref_np = tas_ref.TS.to_numpy()

tas_ref_2d = reshape_3d_to_2d(tas_ref_np)

# Weights based on latitude
[xx,yy] = np.meshgrid(tas_ref.lon,tas_ref.lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# conflevels -> 1st column secular, 2nd column non secular 
print(f'Confidence Interval Calculation for ref sim ({niter} iterations)...')
[conffreq, conflevels] = mtm_svd_conf(tas_ref_2d,nw,kk,dt,niter,sl,w)

# =============================================================================
# save results
# =============================================================================
name = f'conf_int_{case}_run{run}'
name = name + f'_{year_i:04}-{year_f:04}'if chunk else name
name = name + '_NA' if NA else name
name = name + '_unforc' if unforced else name

path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')
np.save(path+name+'.npy',conflevels)

#%% MTM_SVD analysis of ensemble mean -> forced signal

# =============================================================================
# ensemble mean dictionary 
# =============================================================================
CESM_LME_EM = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        CESM_LME_EM[case]  = case_ds.mean(dim = 'run')


# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

NA = False
chunk = False

#chunk years
year_i = 851
year_f = 1150

# =============================================================================
# loop through all cases
# =============================================================================

LFV = {}

dat = CESM_LME_EM

for case_i, ds_i in dat.items():
    
    print('case='+case_i)
    
    ds_i = ds_i.sel(lat = slice(-60,60))
    ds_i = ds_i.where(NA_mask == 1) if NA else ds_i
    
    # Weights based on latitude
    [xx,yy] = np.meshgrid(ds_i.lon,ds_i.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

    # temp data
    if chunk:
        tas = ds_i.sel(year=slice(year_i,year_f))
    else:
        tas = ds_i.isel(year=slice(1,-1))
    tas_np = tas.TS.to_numpy()

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

# merge all dataArrays into a single dataset
lfv = xr.Dataset(LFV)
lfv.attrs["period"] = f'{year_i}-{year_f}'

name = 'lfv_EM'
name = name + f'_{year_i:04}-{year_f:04}'if chunk else name
name = name + '_NA' if NA else name
name = name + '_unforc' if unforced else name

#save results dataset as .nc file
path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')
lfv.to_netcdf(path + name +'.nc')
