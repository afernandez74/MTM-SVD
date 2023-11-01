# Python code that pre-processes the .nc files from all the CMIP6 past1000 simulations
# of the last millennium and locally saves a python dictionary that contains
# only the data to be analyzed by the mtm-svd script (mtm_CESM_LME.py)

# %% import functions and packages

from mtm_funcs import *
from readin_funcs_past1000 import *
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
path = "//Volumes//AlejoED//Work//MannSteinman_Proj//Data//past1000_data//"
files = listdir(path)   
files.sort()

print('Load in data from NetCDF files...')

# read in .nc files and collect lat, lon, sim_number, time and temperature fields 
# put those fields into a dictionary indexed by simulation number and simulation years
[dic_past1000] = nc_to_dic_past1000(path)

# %%-----------------
# 2) Merge entries corresponding to same simulations into single dictinoary entries
# -------------------
    
# merge dictionary entries that correspond to the same simulations
# organize simulation data (temperature and time)
print('Merge files into single simulations...')
dic_CESM_merged = dic_sim_merge_CESM(dic_CESM, sim_no)

# delete unnecessary dictionaries to free memory
del dic_CESM

# %%-----------------
# 3) Calculate annual means
# -------------------
print('Calculate annual means...')
dic_CESM_merged_annual = calc_annual_means_CESM(dic_CESM_merged)
del dic_CESM_merged

# %%-----------------
# 4) Save python dictionary to local directory
# -------------------

save_path = os.path.expanduser("~/mtm_local/CESM_LME_data_dic/")
timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
file_name = f'CESM_LME_data_dic_{timestamp}'
full_path = save_path + file_name

# save data into local directory
with open(full_path,'wb') as f:
    pkl.dump(dic_CESM_merged_annual, f)
    
print (f'Pickle file saved to: {full_path}')
