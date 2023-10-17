#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:29:13 2023

@author: afer
"""
# functions for the reading and pre-processing of the CESM LME simulation files. 


import xarray as xr
import pickle as pkl
from mtm_funcs import *
import numpy as np
import cftime
import os 

## ____________________________________________________________________________

#%%
# create class "sim" which contains information of each simulation file 
class sim:
    def __init__(self, name, sim_no, tas, time, lat, lon):
        self.name = name
        self.sim_no = sim_no
        self.tas = tas
        self.time = time
        self.lat = lat
        self.lon = lon
        
## ____________________________________________________________________________

#%%
def rename_files(path):
    """
    Renames .nc files in a past1000 model simulation so they're easier to handle
    """
    
    #.nc files in the model folder
    models = os.listdir(path)
    models.sort()
    models = [entry for entry in models if not entry.startswith('.')]
    
    # year values are in front of ".nc" in all strings
    id_x = ".nc"
    #length of years string in file names
    len_years = 13 
    
    for model in models:
        path_model = os.path.join(path, model + '/')
        
        files = os.listdir(path_model)
        files.sort()
        files = [entry for entry in files if not entry.startswith('.')]
        
        for file_name in files:
            ix = file_name.find(id_x)
            years = file_name[ix-len_years : ix]
            new_file_name = "tas_" + model + "_" + years + id_x
            old_file = os.path.join(path_model, file_name)
            new_file = os.path.join(path_model, new_file_name)
            os.rename(old_file,new_file)
#%%
def read_in_past1000(path):
    """
    Reads .nc CMIP6 past1000 files and concatenates arrays so that there's 
    one xarray dataset per model. Also calculates annual means 
    """

    # each file is a folder contatining all the .nc files for each model 
    files = os.listdir(path)
    files.sort()
    files = [entry for entry in files if not entry.startswith('.')]
        
    # how many models
    num_models = len(files)
    
    # dictionary
    dicti = {}
    
    for model_name in files:
        print(model_name+'...')
        
        path_model = os.path.join(path,model_name + '/')
        list_files = [path_model + entry for entry in os.listdir(path_model) if not entry.startswith('.')]
        
        ds_monthly = xr.open_mfdataset(list_files, combine='nested', concat_dim='time', use_cftime=True)
        
        # fix error with 'MIROC-ES2L' data where 'time' types are bad for the first year
        if model_name == 'MIROC-ES2L':
            ds_monthly = ds_monthly.isel(time=slice(11,-1))

        # calculate annual means of monthly data
        ds_annual = ds_monthly.groupby('time.year').mean(dim = 'time')
        dicti[model_name] = ds_annual
            
            
    return dicti


## ____________________________________________________________________________

#%%
def convert_to_proleptic_gregorian(dataset):
    datetime360day_data = dataset['time']
    datetime64_data = xr.coding.times.decode_cf_datetime(datetime360day_data,units='')
    dataset['time'] = datetime64_data
    return dataset

#%%
def dic_sim_merge_CESM(dicti, sim_no):
    """
    load the dictionary created with nc_to_dic_CESM and merges the 
    monthly values from each of the 13 simulations
    Return a dictionary in which each entry is a CESM LME simulation
    """
    keys = list(dicti.keys()) # list of keys from old dictionary
    CESM_merged = {}
    lat = dicti[keys[0]].lat
    lon = dicti[keys[0]].lon

    for i in np.unique(sim_no):
    
        
        nam = "".join(["sim",i]) # new key for dictionary w/o dates (eg Sim001)
        print(nam) 
        
        find_str = "".join(['_',i,'_']) # location of the sim number
        
        # keys that contain the desiered simulation in the loop (1-13)
        str_inds = [index for index, s in enumerate(keys) if find_str in s] 
        
        # values of the keys that correspond to the ith simulation
        str_nams = keys[str_inds[0]:str_inds[-1]+1]
        
        tas = np.concatenate([dicti[key].tas for key in str_nams],axis=0)
        time = np.concatenate([dicti[key].time for key in str_nams])
            
        CESM_merged[nam] = sim(nam,i,tas,time,lat,lon)
        
    return CESM_merged
        
## ____________________________________________________________________________

#%%
def calc_annual_means_CESM(dicti):
    """
    Read in a dictionary containing single simulations and their data per value-key pair
    And assign annual means of the data to a similar dictionary of the same format
    """
    keys = list(dicti.keys()) # list of keys from old dictionary
    CESM_merged_annual = {} 
    lat = dicti[keys[0]].lat
    lon = dicti[keys[0]].lon
    
    for i_key in keys:
        tas = dicti[i_key].tas
        time = dicti[i_key].time
        tas_annual, time_years= annual_means_3d(tas,time)
        CESM_merged_annual[i_key] = sim(dicti[i_key].name,dicti[i_key].sim_no,tas_annual,time_years,lat,lon)
        print (dicti[i_key].name)
    return CESM_merged_annual
    
