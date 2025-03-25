import pickle
import pandas as pd
import scipy.io
import os 
import numpy as np
import xarray as xr
import lipd
import matplotlib.pyplot as plt
from proxy_import_funcs import lipd2df
from proxy_import_funcs import annualize_data
from proxy_import_funcs import mask_data 

#%% pages2k summary table
path = os.path.expanduser('~/mtm_local/proxy_data/fernandez_comp/metadata/')
summary_table = pd.read_csv(path+'pages2k_summary_table.csv')

#%% 1)filter proxies with criteria
df = summary_table

start_year = 1600
end_year = 1850
min_resolution = 10

df['min_year_int'] = df['min_year']#.apply(lambda x: -int(x[1:]) if x.startswith('−') else int(x))
start_good = df['min_year_int'] <= start_year

df['max_year_int'] = df['max_year']
end_good = df['max_year_int'] >= end_year

df['resolution_int'] = df['resolution'].apply(lambda x: int(x) if not x.startswith('<') else 0)
resolution_good = df['resolution_int'] <= min_resolution

in_comps_good = df['in_comps'] == 'yes'

all_good = start_good & end_good & resolution_good & in_comps_good

df_good = df[all_good]
#make sure passes composites thing
del start_good, end_good, resolution_good, all_good, df

#%% 2) read in ALL lipd files

path = os.path.expanduser("~/mtm_local/pages2k/LiPD_Files/")
files = os.listdir(path)
files = [entry for entry in files if not entry.startswith('.')]

df_all = lipd2df(path,os.path.expanduser('~/mtm_local/pages2k/LiPD_py_dic/df_all'))

with open (os.path.expanduser('~/mtm_local/pages2k/LiPD_py_dic/df_all'), 'rb') as f:
    df_all = pickle.load(f)
del path, files
#%% 3) filter lipd files based on criteria

df_filtered = df_all[df_all['paleoData_pages2kID'].isin(df_good['PAGES_ID'])]
df_filtered['resolution'] = df_filtered['paleoData_pages2kID'].map(df_good.set_index('PAGES_ID')['resolution_int'])
df_filtered['site_name'] = df_filtered['paleoData_pages2kID'].map(df_good.set_index('PAGES_ID')['site_name'])
df_filtered['direction'] = df_filtered['paleoData_pages2kID'].map(df_good.set_index('PAGES_ID')['direction'])
df_filtered['AMV'] = df_filtered['paleoData_pages2kID'].map(df_good.set_index('PAGES_ID')['AMV'])
df_filtered['in_comps'] = df_filtered['paleoData_pages2kID'].map(df_good.set_index('PAGES_ID')['in_comps'])
df = df_filtered
del df_filtered#, df_good

#%% 4) interpolate to annual resolution and get rid of data outside range

# fix nans 
for ix,row in df.iterrows():
    # print(row)
    if any(isinstance(x, str) for x in row['paleoData_values']):
        dat = row['paleoData_values']
        dat = [np.nan if x == "nan" else x for x in dat]
        df.at[ix,'paleoData_values'] = dat
    

# interpolate to annual resolution and flip descending timeseries
df_interp = df.apply(annualize_data, axis = 1)

# apply filtering to get rid of data outside time range
df_interp = df_interp.apply(mask_data, axis=1, min_year=start_year, max_year=end_year)

#%% 2) transform data into xarray datasets and save as .nc files
for index, row in df_interp.iterrows():
    
    # data for dimensions (year) and data (values)
    year = row['year']
    values = row['paleoData_values']
    
    # attributes for dataarrays
    proxy_type = row['paleoData_proxy']
    proxy_units = row['paleoData_units']
    variable_name = row['paleoData_variableName']
    site_name = row['site_name']
    archive_type = row['archiveType']
    data_set_name = row['dataSetName']
    pages2kID = row['paleoData_pages2kID']
    lat = row['geo_meanLat']
    lon = row['geo_meanLon']
    elev = row['geo_meanElev']
    direction = row['direction']
    AMV = row['AMV']

    # create xarray dataarrays
    year_coord = xr.DataArray(year, dims='time', coords={'time': year}, attrs={'units': 'year AD'})
    values_array = xr.DataArray(values, dims='time', coords={'time': year_coord}, attrs = {'units':archive_type +'_'+variable_name})
                              
    attrs_dic={'proxy_type':proxy_type,'units': proxy_units, 'site_name':site_name,
    'variable_name':variable_name,'archive_type':archive_type,
    'data_set_name':data_set_name,'pages2kID':pages2kID,
    'lat':lat, 'lon':lon,'elev':elev, 'direction':direction,
    'AMV':AMV
    }

    # create xarray dataset
    ds_i = xr.Dataset({'proxy_data': values_array})
    ds_i.attrs = attrs_dic

    # save .nc files
    save_path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets/datasets{start_year}_{end_year}/")
    file_name = f'{pages2kID}_{data_set_name}.nc'
    full_path = save_path + file_name
    ds_i.to_netcdf(full_path)

print (f'Datasets saved to: {save_path}')

