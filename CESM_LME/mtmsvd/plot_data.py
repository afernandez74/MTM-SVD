#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:53:29 2024

@author: afer
"""
# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyleoclim as pyleo
#%% 
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

path = os.path.expanduser("~/mtm_local/CESM_LME/data/")

cases = [entry for entry in os.listdir(path) if not entry.startswith('.')]

CESM_LME = {}
for case in cases:
    path_case = path+case+'/'
    print(path_case)
    file = [entry for entry in os.listdir(path_case) if not entry.startswith('.')][0]
    ds = xr.open_mfdataset(path_case+file)
    CESM_LME[case] = ds.TS
    
CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean

# load North Atlantic mask file 

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)
#%% create pyleoclim ensemble series objects 

NA = True

# year_i = 851
year_i = 851
year_f = 1849

lats = slice(-90,90)


# weights by latitude 
weights = np.cos(np.deg2rad(CESM_LME['ALL_FORCING'].lat))
weights.name = "weights"

# =============================================================================
# ALL FORCING ensemble
# =============================================================================
case = 'ALL_FORCING'

data = CESM_LME[case]
if NA:
    data = data.where(NA_mask == 1)
else:
    data = data.sel(lat = lats)
    
data = data.sel(year = slice(year_i,year_f))
data = data.weighted(weights).mean(dim = ['lat','lon'])

series_ALL = {}
for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_ALL[f'{run:03}'] = series

em = data.mean(dim = ['run'])
serie_ALL_em  = pyleo.Series(
        time=em.year,
        value=em.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label='AF_em')

series_ALL_mult = pyleo.MultipleSeries(list(series_ALL.values()))
series_ALL_ensemb = pyleo.EnsembleSeries(list(series_ALL.values()))
# =============================================================================
# ALL FORCING unforced ensemble
# =============================================================================
case = 'ALL_FORCING'

data = CESM_LME_unforced[case]

if NA:
    data = data.where(NA_mask == 1)
else:
    data = data.sel(lat = lats)

    
data = data.sel(year = slice(year_i,year_f))
data = data.weighted(weights).mean(dim = ['lat','lon'])

series_ALL_unf = {}
for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_ALL_unf[f'{run:03}'] = series

series_ALL_mult_unf = pyleo.MultipleSeries(list(series_ALL_unf.values()))
series_ALL_ensemb_unf = pyleo.EnsembleSeries(list(series_ALL_unf.values()))

#%% NA temperature anomalies figure AF_ONLY

filter_per=20
series_ALL_ensemb_anom = pyleo.EnsembleSeries([])
for series in series_ALL_ensemb.series_list:
    anom = series.center()
    series_ALL_ensemb_anom.series_list.append(anom)

fig, ax = plt.subplots(figsize = [20,10])
series_ALL_ensemb_anom.filter(cutoff_freq=1/filter_per).plot_envelope(ax = ax,
                                                              shade_alpha = 0.1,
                                                              curve_lw = 0,shade_clr='red')
serie_ALL_em.center().filter(cutoff_freq=1/filter_per).plot(ax = ax,linewidth=3,color = 'black')
# series_ALL_ensemb.stripes()

# series_ALL_ensemb_unf.filter(cutoff_freq=1/30).plot_envelope(ax = ax,
#                                                               shade_alpha = 0.1,
#                                                               curve_lw = 0,shade_clr='blue')

plt.xlabel('Year')
plt.ylabel("Mean Global Surface Air Temperature Anomaly")
if NA:
    plt.ylabel("North Atlantic Surface Air Temperature Anomaly")
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
name = f'gmsat_{filter_per}lowpass_AF_ONLY'
name = name + '_NA' if NA else name

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% calculations

means = {}
means_smooth = {}

filter_per = 20

NA = True

for key, value in CESM_LME.items():
    if NA:
        value = value.where (NA_mask==1)
    else:
        value = value.sel(lat = slice(-90,90))

    # if key == 'ALL_FORCING': # 'ALL_FORCING' contains ensemble mean, which we don't want in the mean calculation
    #     means[key] = value.isel(run = slice(0,-1)).mean(dim=['run','lat','lon'])

    #     means_filtered = butter_lowpass(
    #         means[key].isel(year=slice(filter_per,-filter_per)), cutoff_frequency = 1/filter_per, sampling_frequency = 1)

    #     means_smooth[key] = xr.DataArray(
    #         data = means_filtered,
    #         dims = means[key].dims,
    #         coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per,-filter_per))})

    if key == 'CNTL':
        means[key] = value.mean(dim=['lat','lon'])

        means_filtered = butter_lowpass(
            means[key].isel(year=slice(filter_per,-filter_per)), cutoff_frequency = 1/filter_per, sampling_frequency = 1)

        means_smooth[key] = xr.DataArray(
            data = means_filtered,
            dims = means[key].dims,
            coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per,-filter_per))})

    else:
        means[key] = value.mean(dim=['run','lat','lon'])

        means_filtered = butter_lowpass(
            means[key].isel(year=slice(filter_per,-filter_per)), cutoff_frequency = 1/filter_per, sampling_frequency = 1)
        

        means_smooth[key] = xr.DataArray(
            data = means_filtered,
            dims = means[key].dims,
            coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per,-filter_per))})

#%% global temperature anomalies figure

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

ax.grid(True,which='major',axis='both')
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# plot lines

case = 'ALL_FORCING'

p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=1,
        zorder = 2,
        color = 'blue',
        alpha = 0.3,
        label = case)
p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'blue',
        label = case)

case = 'VOLC'

p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=1,
        zorder = 2,
        color = 'red',
        alpha = 0.3,
        label = case)
p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'red',
        label = case)

# case = 'SOLAR'

# p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'gold',
#         alpha = 0.3,
#         label = case)
# p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'gold',
#         label = case)

# case = 'CNTL'

# p2 = means[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'darkgrey',
#         alpha = 0.3,
#         label = case)
# p3 = means_smooth[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'darkgrey',
#         label = case)

# case = 'ORBITAL'

# p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'lightskyblue',
#         alpha = 0.3,
#         label = case)
# p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'lightskyblue',
#         label = case)

plt.xlabel('Year')
plt.ylabel("Mean Global Surface Air Temperature Anomaly")
if NA:
    plt.ylabel("North Atlantic Surface Air Temperature Anomaly")
# plt.title('CESM LME simulations')
ax.set_xlim([900,1800])
ax.legend()
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
name = f'gmsat_{filter_per}lowpass_LM'
name = name + 'NA' if NA else name

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% plot single timeseries


# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

ax.grid(True,which='major',axis='both')
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# plot lines

case = 'ALL_FORCING'

p1 = means[case].isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=1,
        zorder = 2,
        color = 'darkblue',
        alpha = 0.3,
        label = case)
p2 = means_smooth[case].isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'darkblue',
        label = case)

case = 'VOLC'

p2 = means[case].isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=1,
        zorder = 2,
        color = 'darkred',
        alpha = 0.3,
        label = case)
p3 = means_smooth[case].isel(year=slice(0,-1)).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'darkred',
        label = case)

# case = 'SOLAR'

# p2 = means[case].isel(year=slice(0,-1)).TS.plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'orange',
#         alpha = 0.3,
#         label = case)
# p3 = means_smooth[case].isel(year=slice(0,-1)).TS.plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'orange',
#         label = case)

# case = 'CNTL'

# p2 = means[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'darkgrey',
#         alpha = 0.3,
#         label = case)
# p3 = means_smooth[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'darkgrey',
#         label = case)

# case = 'ORBITAL'

# p2 = means[case].isel(year=slice(0,-1)).TS.plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'magenta',
#         alpha = 0.3,
#         label = case)
# p3 = means_smooth[case].isel(year=slice(0,-1)).TS.plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'magenta',
#         label = case)

plt.xlabel('Year')
plt.ylabel("Mean Global Surface Air Temperature")
plt.title('CESM LME simulations')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
name = f'single_gmsat_{filter_per}lowpass_LM'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% plot band pass 30-100 year data
means = {}
means_smooth = {}

filter_per_hi = 30
filter_per_lo = 100

NA = False

for key, value in CESM_LME.items():
    if NA:
        value = value.where (NA_mask==1)
    else:
        value = value.sel(lat = slice(-90,90))

    # if key == 'ALL_FORCING': # 'ALL_FORCING' contains ensemble mean, which we don't want in the mean calculation
    #     means[key] = value.isel(run = slice(0,-1)).mean(dim=['run','lat','lon'])

    #     means_filtered = butter_lowpass(
    #         means[key].isel(year=slice(filter_per,-filter_per)), cutoff_frequency = 1/filter_per, sampling_frequency = 1)

    #     means_smooth[key] = xr.DataArray(
    #         data = means_filtered,
    #         dims = means[key].dims,
    #         coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per,-filter_per))})

    if key == 'CNTL':
        means[key] = value.mean(dim=['lat','lon'])

        means_filtered = butter_bandpass(
            means[key].isel(year=slice(filter_per_hi,-filter_per_hi)), low_cutoff_frequency = 1/filter_per_lo, high_cutoff_frequency = 1/filter_per_hi, sampling_frequency = 1)

        means_smooth[key] = xr.DataArray(
            data = means_filtered,
            dims = means[key].dims,
            coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per_hi,-filter_per_hi))})

    else:
        means[key] = value.mean(dim=['run','lat','lon'])

        means_filtered = butter_bandpass(
            means[key].isel(year=slice(filter_per_hi,-filter_per_hi)), low_cutoff_frequency = 1/filter_per_lo, high_cutoff_frequency = 1/filter_per_hi, sampling_frequency = 1)
        

        means_smooth[key] = xr.DataArray(
            data = means_filtered,
            dims = means[key].dims,
            coords = {'year':CESM_LME[case].year.isel(year = slice(filter_per_hi,-filter_per_hi))})

# plot

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

ax.grid(True,which='major',axis='both')
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# plot lines

case = 'ALL_FORCING'

p2 = (means_smooth[case]).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'blue',
        label = case)

case = 'VOLC'

p2 = (means_smooth[case]).plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'red',
        label = case)

# case = 'SOLAR'

# p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'gold',
#         alpha = 0.3,
#         label = case)
# p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'gold',
#         label = case)

# case = 'CNTL'

# p2 = means[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'darkgrey',
#         alpha = 0.3,
#         label = case)
# p3 = means_smooth[case].isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'darkgrey',
#         label = case)

# case = 'ORBITAL'

# p1 = (means[case]-means[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=1,
#         zorder = 2,
#         color = 'lightskyblue',
#         alpha = 0.3,
#         label = case)
# p2 = (means_smooth[case]-means_smooth[case].mean()).isel(year=slice(0,-1)).plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'lightskyblue',
#         label = case)

plt.xlabel('Year')
plt.ylabel("Mean Global Surface Air Temperature Anomaly")
if NA:
    plt.ylabel("North Atlantic Surface Air Temperature Anomaly")
# plt.title('CESM LME simulations')
ax.set_xlim([850,1850])
ax.legend()
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
name = f'gmsat_{filter_per_hi}-{filter_per_lo}bandpass_LM'
name = name + 'NA' if NA else name

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()