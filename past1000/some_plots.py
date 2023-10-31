
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
import time

# %% Load dictionaries with data
# -------------------

# =============================================================================
# open regular tas datasets dictionary
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000 = pkl.load(f)

del file_path, files, f

# =============================================================================
# open ensemble mean of all past1000 datasets
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic_regrid_ensemb_mean/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_ensemb_mean = pkl.load(f)
    
del file_path, files, f

# =============================================================================
# open unforced dictionary
# =============================================================================
file_path = os.path.expanduser("~/mtm_local/past1000/tas_dic_regrid_unforced/")

# obtain name of file (only works if one file is present only)
files = os.listdir(file_path)
files.sort()
files = [entry for entry in files if not entry.startswith('.')]

with open(file_path + files[0], 'rb') as f:
    past1000_unforced = pkl.load(f)
    
del file_path, files, f


#%%plot global mean Temperature anomalies for all models

fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

for key, ds_i in past1000.items():
    (ds_i.mean(dim=['lat','lon']).tas - (ds_i.mean(dim=['lat','lon','year']).tas)).plot(label=key, linewidth = 0.5, alpha = 0.8)

(past1000_ensemb_mean.mean(dim=['lat','lon']).tas - past1000_ensemb_mean.mean(dim=['lat','lon','year']).tas).plot(label='ensemble mean', color = 'black', linewidth=1.5)
ax.grid(True,which='major',axis='both')
plt.title("Temperature anomalies CMIP6 past1000" )
plt.legend()


#%% plot forced timeseries only (ensemble mean)
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
past1000_ensemb_mean.mean(dim=['lat','lon']).tas.plot(label='ensemble mean', color = 'black', linewidth=1.5)

#%% plot unforced timeseries (past1000 - ensemble mean)

fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

for key, ds_i in past1000_unforced.items():
    ts_raw = ds_i.mean(dim=['lat','lon']).tas
    ts_mean = np.mean(ts_raw)
    ts_std = np.std(ts_raw)
    ts_plot = (ts_raw-ts_mean)/(ts_std)
    ax.plot(ds_i.year,ts_plot, label = key, linewidth = 0.5, alpha = 0.9)
    
past1000_unforced_concat = xr.concat(list(past1000_unforced.values()), dim='model', coords = 'minimal')
past1000_unforced_concat_mean = past1000_unforced_concat.mean(dim=['lat','lon','model'])
ax.plot(ds_i.year,past1000_unforced_concat_mean.tas, label = 'unforced timeseries mean', color = 'black', linewidth = 1.5)
plt.legend()