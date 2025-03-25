#import records compiled by Fernandez et al., 2025

import pickle
import pandas as pd
import scipy.io
import os 
import numpy as np
import xarray as xr
import lipd
import matplotlib.pyplot as plt
from proxy_import_funcs import lipd2df, annualize_data, annualize_data_AF, mask_data

#%% pages2k summary table
path = os.path.expanduser('~/mtm_local/proxy_data/fernandez_comp/metadata/')
p2k_summary_table = pd.read_csv(path+'pages2k_summary_table.csv')

# Fernandez summary table
AF_summary_table = pd.read_csv(path+'fernandez_summary_table.csv')

#drop records whose data has not been added to the 'Usables' folder
AF_archives = AF_summary_table.drop(AF_summary_table[AF_summary_table.Usables!='Yes'].index)

AF_archives['resolution_int'] = AF_archives['resolution'].apply(lambda x: int(x) if not x.startswith('<') else 0)
#%% Read csv files of data and create xarray datasets
path = os.path.expanduser('~/mtm_local/proxy_data/fernandez_comp/usables/')
files = os.listdir(path)
files = [entry for entry in files if not entry.startswith('.')]
#%% create xarray datasets and save to the appropriate directories
for ix, arch in AF_archives.iterrows():

    rez = arch.resolution_int
    name = arch.PAGES_ID
    print(name)
    file = [i for i in files if name in i][0]
    
    # attributes for dataarrays
    proxy_type = arch['Archive']
    proxy_units = arch['units']
    variable_name = arch['variable']
    site_name = arch['site_name']
    archive_type = arch['Archive']
    data_set_name = name
    pages2kID = name
    lat = arch['lat']
    lon = arch['lon']
    elev = np.nan
    direction = arch['Direction']
    AMV = arch['AMV']

    dat = pd.read_csv(path + file, dtype={"proxy_data": float}, na_values = 'NA')
    
    # interpolate to annual, flip reverse timeseries and/or calc annual means if 
    # finer than annual resolution
    dat_annual = annualize_data_AF(dat, rez)
    
    #drop data after 1850
    dat_annual = dat_annual.drop(dat_annual[dat_annual.Age_CE>1850].index)
    # drop data before one of the cutoff years
    if dat_annual.Age_CE[0] < 850:
        dat_annual = dat_annual.drop(dat_annual[dat_annual.Age_CE<850].index)
        start_yr = 850
        
    elif dat_annual.Age_CE[0] <= 1200:
        dat_annual = dat_annual.drop(dat_annual[dat_annual.Age_CE<1200].index)
        start_yr = 1200
        
    elif dat_annual.Age_CE[0] < 1400:
        dat_annual = dat_annual.drop(dat_annual[dat_annual.Age_CE<1400].index)
        start_yr = 1400
   
    elif dat_annual.Age_CE[0] <= 1600:
        dat_annual = dat_annual.drop(dat_annual[dat_annual.Age_CE<1600].index)
        start_yr = 1600

    # create xarray dataarrays
    year = dat_annual['Age_CE']
    values = dat_annual['proxy_data']

    
    year_coord = xr.DataArray(year, dims='time', coords={'time': year}, attrs={'units': 'year AD'})
    values_array = xr.DataArray(values, dims='time', coords={'time': year_coord}, attrs = {'units':archive_type +'_'+variable_name})
                              
    attrs_dic={'proxy_type':proxy_type,'units': proxy_units, 'site_name':site_name,
    'variable_name':variable_name,'archive_type':archive_type,
    'data_set_name':data_set_name,'pages2kID':pages2kID,
    'lat':lat, 'lon':lon,'elev':elev, 'direction':direction, 'AMV':AMV}

    # create xarray dataset
    ds_i = xr.Dataset({'proxy_data': values_array})
    ds_i.attrs = attrs_dic

    # save .nc files
    
    if start_yr == 850:
        save_path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets/datasets{start_yr}_1850/")
        file_name = f'AF_{data_set_name}.nc'
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)
        
        ds_i = ds_i.sel(time=slice(1200,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1200_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)
        
        ds_i = ds_i.sel(time=slice(1400,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1400_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)

        ds_i = ds_i.sel(time=slice(1600,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1600_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)   
        
    elif start_yr == 1200:
        save_path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets//datasets{start_yr}_1850/")
        file_name = f'AF_{data_set_name}.nc'
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)

        ds_i = ds_i.sel(time=slice(1400,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1400_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)

        ds_i = ds_i.sel(time=slice(1600,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1600_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)   

    elif start_yr == 1400:
    
        save_path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets//datasets{start_yr}_1850/")
        file_name = f'AF_{data_set_name}.nc'
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)
    
        ds_i = ds_i.sel(time=slice(1600,1850))
        save_path = os.path.expanduser("~/mtm_local/proxy_data/compiled_datasets//datasets1600_1850/") 
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)   
        
    else:
        save_path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets//datasets{start_yr}_1850/")
        file_name = f'AF_{data_set_name}.nc'
        full_path = save_path + file_name
        ds_i.to_netcdf(full_path)
        
    
    
    
    
    