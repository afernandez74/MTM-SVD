#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 17:54:29 2023

@author: afer
"""
import os 
import lipd
import pandas as pd
import numpy as np

def lipd2df(lipd_dirpath, pkl_filepath, col_str=[
            'paleoData_pages2kID',
            'dataSetName', 'archiveType',                                                                                
            'geo_meanElev', 'geo_meanLat', 'geo_meanLon',
            'year', 'yearUnits',                                                                                         
            'paleoData_variableName',
            'paleoData_units',                                                                                           
            'paleoData_values',
            'paleoData_proxy']):
    
    ''' Convert a bunch of PAGES2k LiPD files to a pickle file of Pandas DataFrame to boost data loading                 
                                                   
    Args:                                          
        lipd_dirpath (str): the path of the PAGES2k LiPD files
        pkl_filepath (str): the path of the converted pickle file
        col_str (list of str): the name string of the variables to extract from the LiPD files
    
    Returns:                                                         
        df (Pandas DataFrame): the converted Pandas DataFrame
    '''
    
    # save the current working directory for later use, as the LiPD utility will change it in the background
    work_dir = os.getcwd()
    
    # LiPD utility requries the absolute path, so let's get it
    lipd_dirpath = os.path.abspath(lipd_dirpath)
    
    # load LiPD files from the given directory
    lipds = lipd.readLipd(lipd_dirpath)
    
    # extract timeseries from the list of LiDP objects
    ts_list = lipd.extractTs(lipds)
    
    # recover the working directory
    os.chdir(work_dir)
    
    # create an empty pandas dataframe with the number of rows to be the number of the timeseries (PAGES2k records),
    # and the columns to be the variables we'd like to extract 
    df_tmp = pd.DataFrame(index=range(len(ts_list)), columns=col_str)
    
    # loop over the timeseries and pick those for global temperature analysis
    i = 0                                                                                                                
    for ts in ts_list:
        if 'paleoData_useInGlobalTemperatureAnalysis' in ts.keys() and \
            ts['paleoData_useInGlobalTemperatureAnalysis'] == 'TRUE':
            for name in col_str:                                                                                         
                try:
                    df_tmp.loc[i, name] = ts[name]                                                                       
                except:
                    df_tmp.loc[i, name] = np.nan                                                                         
            i += 1 
            
    # drop the rows with all NaNs (those not for global temperature analysis)
    df = df_tmp.dropna(how='all')
    
    # save the dataframe to a pickle file for later use
    save_path = os.path.abspath(pkl_filepath)
    print(f'Saving pickle file at: {save_path}')
    df.to_pickle(save_path)
    
    return df

# Function to interpolate 'data' values for each dataset from PAGES2k
# also flips descending timeseries
def annualize_data(row):
        
    # get rid of NaN values beforehand
    if any(np.isnan(x) for x in row['paleoData_values']):     
        nan_indices = [i for i, x in enumerate(row['paleoData_values']) if np.isnan(x)]
        row['paleoData_values'] = [x for i, x in enumerate(row['paleoData_values']) if i not in nan_indices]
        row['year'] = [x for i, x in enumerate(row['year']) if i not in nan_indices]
    
    year_array = np.array(row['year'])
    data_array = np.array(row['paleoData_values'])
    
    name = row['paleoData_pages2kID']
    print (name)
    
    if not monotonic(year_array):
        print('not monotonic')
        if year_array[0] < year_array[-1]: #ascending
            year_array = fix_non_increasing_values(year_array)
        else:
            year_array = fix_non_decreasing_values(year_array)
    # if already annual resolution, only flip descending timeseries and return row
    if row['resolution'] == 1:
        if all(x >= y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)
        new_years = [int(year) for year in year_array]
        row['year'] = new_years
        row['paleoData_values'] = data_array

        return row
    # if coarser than annual resolution, interpolate to annual 
    elif row['resolution'] >= 1:
        if all(x >= y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)

        new_years = np.arange(int(min(year_array)), int(max(year_array)))
        interpolated_data = np.interp(new_years, year_array, data_array)

        row['year'] = [int(year) for year in new_years]
        row['paleoData_values'] = interpolated_data

        return row
    # if finer than annual resolution, compute annual means 
    else:
        if all(x > y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)
        # round year value to int
        new_years = year_array.round(0).astype(int)
        
        # compute the annual means
        row2 = pd.DataFrame({'year':new_years, 'data': data_array})
        row2['annual_data'] = row2.groupby('year')['data'].transform('mean')
        row2_resampled = row2[['year','annual_data']].drop_duplicates()
        
        row['year'] = row2_resampled['year']
        row['paleoData_values'] = row2_resampled['annual_data']
        return row
    
# Function to interpolate data from Fernandez compilation
# also flips descending timeseries
def annualize_data_AF(row,resolution):
    # get rid of NaN values beforehand
    if row.proxy_data.isna().any():
        
        row = row.drop(row[(row.proxy_data.isna())].index)

    year_array = np.array(row['Age_CE'])
    data_array = np.array(row['proxy_data'])
    if np.issubdtype(data_array.dtype, np.str_):
        data_array = data_array.astype(float)
    
    if not monotonic(year_array):
        print('not monotonic')
        if year_array[0] < year_array[-1]: #ascending
            year_array = fix_non_increasing_values(year_array)
        else:
            year_array = fix_non_decreasing_values(year_array)
    # if already annual resolution, only flip descending timeseries and return row
    if resolution == 1:
        if all(x >= y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)
        new_years = [int(year) for year in year_array]
        row['Age_CE'] = new_years
        row['proxy_data'] = data_array

        return row
    
    # if coarser than annual resolution, interpolate to annual 
    elif resolution >= 1:
        if all(x >= y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)

        new_years = np.arange(int(min(year_array)), int(max(year_array)))
        interpolated_data = np.interp(new_years, year_array, data_array)
        
        new_years = [int(year) for year in new_years]
        row2 = pd.DataFrame({'Age_CE':new_years, 'proxy_data': interpolated_data})
        # row['Age_CE'] = [int(year) for year in new_years]
        # row['proxy_data'] = interpolated_data
        row = row2
        return row
    
    # if finer than annual resolution, compute annual means 
    else:
        if all(x > y for x, y in zip(year_array, year_array[1:])): #flip data if descending
            year_array = np.flip(year_array)
            data_array = np.flip(data_array)
        # round year value to int
        new_years = year_array.round(0).astype(int)
        
        # compute the annual means
        row2 = pd.DataFrame({'year':new_years, 'data': data_array})
        row2['annual_data'] = row2.groupby('year')['data'].transform('mean')
        row2_resampled = row2[['year','annual_data']].drop_duplicates()
        
        row['Age_CE'] = row2_resampled['year']
        row['proxy_data'] = row2_resampled['annual_data']
        return row

# filter out proxy data outside desired range
def mask_data(row, min_year, max_year):
    year_array = row['year']
    data_array = row['paleoData_values']

    mask = np.logical_and(np.array(year_array) >= min_year, np.array(year_array) <= max_year)

    # Filter 'year' and 'data' arrays
    row['year'] = np.array(year_array)[mask]
    row['paleoData_values']= np.array(data_array)[mask]

    return row

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

def fix_non_decreasing_values(sequence):
    fixed_sequence = sequence.copy()

    for i in range(1, len(sequence)):
        if sequence[i] >= sequence[i - 1]:
            fixed_sequence[i] = fixed_sequence[i - 1]

    return fixed_sequence

def fix_non_increasing_values(sequence):
    fixed_sequence = sequence.copy()

    for i in range(1, len(sequence)):
        if sequence[i] <= sequence[i - 1]:
            fixed_sequence[i] = fixed_sequence[i - 1]

    return fixed_sequence