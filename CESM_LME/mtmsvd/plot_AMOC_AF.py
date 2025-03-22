import xarray as xr
import os 
import re
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mtmsvd_funcs import butter_lowpass

# modify global setting
SMALL_SIZE = 15
MEDIUM_SIZE = 30
BIGGER_SIZE = 50

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

del BIGGER_SIZE, MEDIUM_SIZE, SMALL_SIZE


#%%
path = os.path.expanduser("~/mtm_local/CESM_LME/MOC_ALL_FORCING_DATA/")
files = os.listdir(path)
var = 'MOC'
transport_reg = 1 #0: global 1: atlantic
# moc_comp = 0 #0 for eulerian (ekman), 1 for eddy, 2 for submeso

# ralization numbers from file names
realizations = [1,2,3,4,5,6,7,8,9,10,11,12,13]

amoc_list = []
# read each realization and obtain timeseries of AMOC
for r in realizations: 
    pat = f'r{r}'
    files_r = [file for file in files if pat in file]
    paths_r = [path + file for file in files_r]
    
    ds = xr.open_mfdataset(paths_r)[var]
    ds = ds.sel(transport_reg=transport_reg)
    ds = ds.sum(dim = 'moc_comp')
    # ds = ds.sel(moc_comp = 0)
    ds = ds.sel(lat_aux_grid = slice(15,85))
    
    #annual means 
    ds = ds.groupby('time.year').mean(dim = 'time', skipna = True)
    
    #Keep only data below 500m depth
    ds = ds.sel(moc_z=ds.moc_z>=50000)

    # obtain maximmum value along the depth dimension
    amoc_ts = ds.max(dim=('moc_z','lat_aux_grid'))
    
    # save amoc realization
    amoc = xr.DataArray(
        data = amoc_ts.values,
        dims = amoc_ts.dims,
        coords = amoc_ts.coords,
        name = pat
        )
    print(pat)
    amoc_list.append(amoc)

amoc = xr.concat(amoc_list, dim=xr.Variable('realization',realizations))

amoc_em = amoc.mean(dim = 'realization')
#%% load volc
path = os.path.expanduser('~/mtm_local/Gao_etal_2008_data/Gao_08_dat_edit.csv')

dat = pd.read_csv(path)

volc= xr.DataArray(dat["Sulf_Tg"].values, coords={"years": dat["Year"]}, dims="years")

volc_roll = volc.rolling(years=100,center=False).mean()

#%%
xlim = [900,1800]

fig, (ax1,ax3) = plt.subplots(2,1,figsize = (30,15))
ax2 = ax1.twinx()

amoc_em_z = (amoc_em-amoc_em.mean())/(amoc_em.std())
amoc_em_filt = (butter_lowpass(amoc_em_z,1/100,1))

# for i in realizations:
#     ix = i-1
#     amoc_i = amoc_list[ix]
#     # mean = amoc_i.mean()
#     # std = amoc_i.std()
#     # z = (amoc_i-mean)/std
#     z = (amoc_i - amoc_em.mean())/(amoc_em.std())
#     z.plot(ax = ax1, linewidth = .5, color='gray')

ax1.plot(amoc_em.year.values, amoc_em_z, color = 'black',linewidth = 2)
ax1.plot(amoc_em.year.values, amoc_em_filt, color = 'gray',linewidth = 2)
ax1.grid(True,which='major',axis='both')
ax1.set_ylabel('AMOC zscore')
ax1.xaxis.set_major_locator(mticker.MultipleLocator(100))
ax1.yaxis.set_major_locator(mticker.MultipleLocator(1))
ax1.set_xlim(xlim)
ax2.plot(amoc_em.year.values, amoc_em_z.rolling(year=100,center=True).std(), color='red')
ax2.set_ylabel('AMOC std dev', color='tab:red')
ax2.tick_params(axis = 'y', labelcolor='tab:red')

ax3.plot(amoc_em.year.values, amoc_em_z, color = None,linewidth = 0)
volc.plot(ax = ax3, color='red')
ax3.set_ylabel('Volcanic Injection [Tg]', color='tab:red')
ax3.tick_params(axis='y', labelcolor='tab:red')
ax3.set_xlim(xlim)
fig.tight_layout()



save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/AMOC/')
name = f'AMOC_volc_comp{xlim[0]}_{xlim[1]}_std'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
