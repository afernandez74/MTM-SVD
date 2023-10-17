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


# %% 1) Rename files for ease of handling

#path to the climate dataset to be utilized
path = "/Volumes/AlejoED/Work/MannSteinman_Proj/Data/past1000_data/"

print (f"Renaming files in {path}")
rename_files(path)

# %% 2) Read in .nc files for all CMIP6 past1000 simulations and concatenate 
#    into single xarray. Also annualizes each dataset and saves into a dictionary
# -------------------

print("Reading in .nc files to a common dictionary")
ds_dic_past1000 = read_in_past1000(path)

# %%-----------------
# 4) Save python dictionary to local directory

save_path = os.path.expanduser("~/mtm_local/CMIP6_past1000_data_dic/")
timestamp = datetime.now().strftime("%b%d_%Y_%I.%M%p")
file_name = f'CMIP6_past1000_data_dic_{timestamp}'
full_path = save_path + file_name

# save data into local directory
with open(full_path,'wb') as f:
    pkl.dump(ds_dic_past1000, f)
    
print (f'Pickle file saved to: {full_path}')
