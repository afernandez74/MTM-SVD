# Python code that pre-processes the .nc files from all the CMIP6 past1000 simulations
# of the last millennium and locally saves a python dictionary that contains
# only the data to be analyzed by the mtm-svd script (mtm_CESM_LME.py)

# %% import functions and packages

from readin_funcs_past1000 import *
import xarray as xr
import os 
from datetime import datetime
import pickle as pkl


# %% 1) Rename files for ease of handling

#path to the climate dataset to be utilized
path = "/Volumes/AlejoED/Work/MannSteinman_Proj/Data/cmip5_past1000_data/"

print (f"Renaming files in {path}")
rename_files(path)

# %% 2) Read in .nc files for all CMIP6 past1000 simulations and concatenate 
#    into single xarray. Also annualizes each dataset and saves into a dictionary
# -------------------

print("Reading in .nc files to a common dictionary")
ds_dic_past1000 = read_in_past1000(path)

# %% 3) Save datasets to local directory

save_path = os.path.expanduser("~/mtm_local/cmip5_past1000/datasets_re_download/")

for model, ds_i in ds_dic_past1000.items():
    print(f"saving {model} data to {save_path}")
    file_name = f'{model}_past1000_dataset.nc'
    full_path = save_path + file_name
    ds_i.to_netcdf(full_path)

print (f'Datasets saved to: {save_path}')
