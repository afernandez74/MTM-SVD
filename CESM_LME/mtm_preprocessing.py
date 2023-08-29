# Python code that pre-processes the .nc files from the CESM_LME simulations
# of the last millennium and locally saves a python dictionary that contains
# only the data to be analyzed by the mtm-svd script (mtm_CESM_LME.py)

# %% import functions and packages

from mtm_functions_AF import *
from read_in_CESM_LME_nc import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl


# %%-----------------
# 1) Load the raw data and save it to dictionary
# -------------------

#path to the climate dataset to be utilized
path = "//Volumes//AlejoED//Work//MannSteinman_Proj//Data//CESM_LME_data//2021_CESM_LME_ALL_FORCING//2021_CESM_LME_ALL_FORCING//"
files = listdir(path)   
files.sort()

print('Load data...')
# read in .nc files and collect lat, lon, sim_number, time and temperature fields 
# put those fields into a dictionary indexed by simulation number and simulation years
[dic_CESM, sim_no] = nc_to_dic_CESM(path)


# %%-----------------
# 2) Merge entries corresponding to same simulations into single dictinoary entries
# -------------------
    
# merge dictionary entries that correspond to the same simulations
# organize simulation data (temperature and time)
dic_CESM_merged = dic_sim_merge_CESM(dic_CESM, sim_no)

# delete unnecessary dictionaries to free memory
del dic_CESM

# %%-----------------
# 3) Calculate annual means
# -------------------
dic_CESM_merged_annual = calc_annual_means_CESM(dic_CESM_merged)
del dic_CESM_merged

# %%-----------------
# 4) Save python dictionary to local directory
# -------------------
