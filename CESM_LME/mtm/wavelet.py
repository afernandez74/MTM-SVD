#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:47:17 2024

@author: afer
"""

import pyleoclim as pyleo
import numpy as np
import xarray as xr
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
pyleo.set_style(style = 'journal',font_scale = 2.0, dpi =300)

#%% Load datasets 

path = os.path.expanduser("~/mtm_local/CESM_LME/data/")

cases = [entry for entry in os.listdir(path) if not entry.startswith('.')]

CESM_LME = {}
for case in cases:
    path_case = path+case+'/'
    print(path_case)
    file = [entry for entry in os.listdir(path_case) if not entry.startswith('.')][0]
    ds = xr.open_mfdataset(path_case+file)
    CESM_LME[case] = ds
    
del case, ds, file, path, path_case

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean

del case, case_ds

# load North Atlantic mask file 

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)
#%%#%% plotting parameters
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
#%% wavelet

# # with waipy:
    
# year_i = 851
# year_f = 1849

# # weights by latitude 
# weights = np.cos(np.deg2rad(CESM_LME['ALL_FORCING'].lat))
# weights.name = "weights"

# NA = True
# case = 'VOLC'
# unforced = False
# ensemble_mean = True
# run = 12

# data = CESM_LME_unforced[case] if unforced else CESM_LME[case] 

# if NA:
#     data = data.where(NA_mask == 1)

# else:
#     data = data.sel(lat = slice(-60,60))

# data = data.sel(year = slice(year_i,year_f)).TS

# if ensemble_mean:
#     data = data.weighted(weights).mean(dim = ['lat','lon','run'])
# else:
#     data = data.weighted(weights).mean(dim = ['lat','lon']).sel(run = run)

# time = data.year.values
# data = data.values

# data_norm = waipy.normalize(data)

# dt = 1
# T1 = 851
# N = len(data_norm)
# pad = 1         # pad the time series with zeroes (recommended)
# dj = 0.25       # this will do 4 sub-octaves per octave
# s0 = 2*dt       # this says start at a scale of 6 months if dt =annual
# j1 = 7/dj       # this says do 7 powers-of-two with dj sub-octaves each
# lag1 = 0.72     # lag-1 autocorrelation for red noise background
# param = 6
# mother = 'Morlet'

# alpha = np.corrcoef(data_norm[0:-1], data_norm[1:])[0,1]
# print("Lag-1 autocorrelation = {:4.2f}".format(alpha))


# result = waipy.cwt(data_norm, dt, pad, dj, s0, j1, alpha, 6, mother='Morlet', name = 'GMSAT')

# #time from waipy.load_txt 
# #data normalize
# dtmin = 0.25/32    # dt/n of suboctaves

# path = os.path.expanduser('~/mtm_local/CESM_LME/figs/wavelets/waipy/')

# if ensemble_mean:
#     name = f'waipy_wavelet_ensemb_mean_{case}'
# else:
#     name = f'waipy_wavelet_run_{run}_{case}'

# name = name + '_NA' if NA else name

# name = name + '_unforc' if unforced else name

# waipy.wavelet_plot(name, time, data_norm, dtmin, result, filename = (path+name+'.svg'))
#%% wavelet with pyleoclim
    
year_i = 851
year_f = 1849

# weights by latitude 
weights = np.cos(np.deg2rad(CESM_LME['ALL_FORCING'].lat))
weights.name = "weights"

NA = True
case = 'ALL_FORCING'
unforced = False
ensemble_mean = True
run = 0

data = CESM_LME_unforced[case] if unforced else CESM_LME[case] 

if NA:
    data = data.where(NA_mask == 1)

else:
    data = data.sel(lat = slice(-90,90))

data = data.sel(year = slice(year_i,year_f)).TS

if ensemble_mean:
    data = data.weighted(weights).mean(dim = ['lat','lon','run'])
else:
    data = data.weighted(weights).mean(dim = ['lat','lon']).sel(run = run)

data_ser = pyleo.Series(
        time=data.year.values,
        value=data.values,
        time_name="years AD",
        time_unit="yr",
        value_name="T",
        value_unit="K")

wave = data_ser.wavelet(
    settings = {'pad':True},
    freq_kwargs={'fmin':1/200,'fmax':1/5,'nf':100}).signif_test(
        method = 'CN',number=1000,qs = [0.5,0.9])

yticks=[100,90,80,70,60,50,40,30,20]

contourf_style = {'cmap': 'turbo', 
                  'origin': 'lower', 
                  'levels': 15}
title = f'scalogram {case}'
title = title + ' Ensemble Mean' if ensemble_mean else title + f'run {run}'
title = title + ' NA' if NA else title
title = title +' Unforced' if unforced else title

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])


fig = wave.plot(
          ylim=[100,10],
          title = title,
          # xlim = [900,1800],
          contourf_style=contourf_style,
          signif_thresh = 0.90,
          signif_clr='black',
          yticks = yticks,
          ax=ax)

path = os.path.expanduser('~/mtm_local/CESM_LME/figs/wavelets/')
path = path + 'NA/' if NA else path + 'GLOB/'

name = title

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(path+name+'.svg', format = 'svg')

else:
    plt.show()


#%% ALL WAVELETS
    
year_i = 851
year_f = 1849

# weights by latitude 
weights = np.cos(np.deg2rad(CESM_LME['ALL_FORCING'].lat))
weights.name = "weights"

NA = True
case = 'ALL_FORCING'
unforced = True
# ensemble_mean = True
# run = 0

data = CESM_LME_unforced[case] if unforced else CESM_LME[case] 

if NA:
    data = data.where(NA_mask == 1)

else:
    data = data.sel(lat = slice(-60,60))

data = data.sel(year = slice(year_i,year_f)).TS
data = data.weighted(weights).mean(dim = ['lat','lon'])

save_fig = input("Save fig? (y/n):").lower()

for run_i in data.run:
    
    data_i = data.sel(run = run_i)
    
    data_ser = pyleo.Series(
            time=data_i.year.values,
            value=data_i.values,
            time_name="years AD",
            time_unit="yr",
            value_name="T",
            value_unit="K")
    
    wave = data_ser.wavelet().signif_test(number=100,qs = [0.5,0.9,0.95])
    
    yticks=[100,80,70,60,50,40,30,20]
    
    contourf_style = {'cmap': 'jet', 
                      'origin': 'lower', 
                      'levels': 100}
    title = f'scalogram {case}'
    title = title + f'run {run_i.values:03}'
    title = title + ' NA' if NA else title
    title = title +' Unforced' if unforced else title
    
    fig = plt.figure(figsize=(30,12))
    ax = fig.add_axes([0.1,0.1,0.5,0.8])
    
    
    fig = wave.plot(ylim=[100,20],
              title = title,
              xlim = [900,1800],
              contourf_style=contourf_style,
              signif_thresh = 0.90,
              signif_clr='black',
              yticks = yticks,
              ax=ax)
    
    path = os.path.expanduser('~/mtm_local/CESM_LME/figs/wavelets/')
    path = path + 'NA/' if NA else path + 'GLOB/'
    
    name = title
    
    if save_fig == 'y':
        plt.savefig(path+name+'.png', format = 'png', dpi = 300)
        plt.savefig(path+name+'.svg', format = 'svg')
    
    else:
        plt.show()
    
