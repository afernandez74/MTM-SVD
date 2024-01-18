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
    files = [entry for entry in files if not entry.startswith('wget')]
    
    # how many models
    num_models = len(files)
    
    # dictionary
    dicti = {}
    
    for model_name in files:
        print(model_name+'...')
        
        path_model = os.path.join(path,model_name + '/')
        list_files = [path_model + entry for entry in os.listdir(path_model) if not entry.startswith('.')]
        
        ds_monthly = xr.open_mfdataset(list_files, combine='nested', concat_dim='time', use_cftime=True)
        
        # calculate annual means of monthly data
        ds_annual = ds_monthly.groupby('time.year').mean(dim = 'time')
        dicti[model_name] = ds_annual
            
    return dicti

