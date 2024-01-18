# Python code that pre-processes the .nc files from all the CMIP6 past1000 simulations
# of the last millennium and locally saves a python dictionary that contains
# only the data to be analyzed by the mtm-svd script (mtm_CESM_LME.py)


# %% import functions and packages

import xarray as xr
import os 

#%% declare paths and read file names for each experiment

# path to CESM LME data 
path = '/Volumes/AlejoED/Work/MannSteinman_Proj/Data/CESM_LME_data/CESM_LME/CESM_LME_tas/'

path_CNTL = path + 'CNTL/'

path_ALL_FORCING = path + 'ALL_FORCING/'

path_VOLC = path + 'VOLC/'

path_ORBITAL = path + 'ORBITAL/'

path_SOLAR = path + 'SOLAR/'


#%%Read .nc files 

CNTL = xr.open_mfdataset(path_CNTL+'*.nc')

ALL_FORCING = xr.open_mfdataset(path_ALL_FORCING+'*.nc', combine = 'nested', concat_dim = 'run')

VOLC = xr.open_mfdataset(path_VOLC+'*.nc', combine = 'nested', concat_dim = 'run')

ORBITAL = xr.open_mfdataset(path_ORBITAL+'*.nc', combine = 'nested', concat_dim = 'run')

SOLAR = xr.open_mfdataset(path_SOLAR+'*.nc', combine = 'nested', concat_dim = 'run')

#%% Calculate annual means

CNTL_annual = CNTL.groupby('time.year').mean(dim='time', skipna=True)

ALL_FORCING_annual = ALL_FORCING.groupby('time.year').mean(dim='time', skipna=True)

VOLC_annual = VOLC.groupby('time.year').mean(dim='time', skipna=True)

ORBITAL_annual = ORBITAL.groupby('time.year').mean(dim='time', skipna=True)

SOLAR_annual = SOLAR.groupby('time.year').mean(dim='time', skipna=True)

#%%Save annualized pre-processed data to local directory

save_path = os.path.expanduser("~/mtm_local/CESM_LME/data/")

CNTL_annual.to_netcdf(save_path+'CNTL/'+'CNTL_annual.nc')

ALL_FORCING_annual.to_netcdf(save_path+'ALL_FORCING/'+'ALL_FORCING_annual.nc')

VOLC_annual.to_netcdf(save_path+'VOLC/'+'VOLC_annual.nc')

ORBITAL_annual.to_netcdf(save_path+'ORBITAL/'+'ORBITAL_annual.nc')

SOLAR_annual.to_netcdf(save_path+'SOLAR/'+'SOLAR_annual.nc')