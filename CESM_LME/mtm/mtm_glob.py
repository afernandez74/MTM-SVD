#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:53:54 2024

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
#%% plotting parameters
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

#%% create pyleoclim ensemble series objects 

# =============================================================================
# ALL FORCING ensemble
# =============================================================================
lats = slice(-60,60)

data = CESM_LME['ALL_FORCING'].sel(lat = lats).mean(dim=['lat','lon'])

series_ALL = {}
for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_ALL[f'{run:03}'] = series


# =============================================================================
# VOlcanic  ensemble
# =============================================================================
data = CESM_LME['VOLC'].sel(lat = lats).mean(dim=['lat','lon'])

series_VOLC={}

for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_VOLC[f'{run:03}'] = series


# =============================================================================
# Solar  ensemble
# =============================================================================
data = CESM_LME['SOLAR'].sel(lat = lats).mean(dim=['lat','lon'])

series_SOLAR={}

for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_SOLAR[f'{run:03}'] = series


# =============================================================================
# Orbital  ensemble
# =============================================================================
data = CESM_LME['ORBITAL'].sel(lat = lats).mean(dim=['lat','lon'])

series_ORBITAL={}

for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_ORBITAL[f'{run:03}'] = series


del data, run, dat, series, 

#%% MTM  psd analysis

# bandwidth parameter
nw = 2
N = len(series_ALL['000'].time)
npad = 2**int(np.ceil(np.log2(abs(N)))+2)

# All forcing
psd_ALL = {}

for run, serie in series_ALL.items():
    psd_ALL[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})
        

# Volcanic
psd_VOLC = {}

for run, serie in series_VOLC.items():
    psd_VOLC[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})

# Solar
psd_SOLAR = {}

for run, serie in series_SOLAR.items():
    psd_SOLAR[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})


# Orbital
psd_ORBITAL = {}

for run, serie in series_ORBITAL.items():
    psd_ORBITAL[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})

#%%Significance tests

sig_n = 1000
qs = [0.50,0.90,0.99]

# All forcing
signif_ALL = psd_ALL['010'].signif_test(number=sig_n, qs=qs)
signif_ALL_series = signif_ALL.signif_qs.psd_list

# Volcanic
signif_VOLC = psd_VOLC['000'].signif_test(number=sig_n, qs=qs)
signif_VOLC_series = signif_VOLC.signif_qs.psd_list

# # Solar
# signif_SOLAR = psd_SOLAR['000'].signif_test(number=sig_n, qs=qs)
# signif_SOLAR_series = signif_SOLAR.signif_qs.psd_list

# # Orbital
# signif_ORBITAL = psd_ORBITAL['000'].signif_test(number=sig_n, qs=qs)
# signif_ORBITAL_series = signif_ORBITAL.signif_qs.psd_list


#%% Get results in numpy arrays (out of Pyleoclim)

# All Forcing
psd_ALL_np = {}
psd_ALL_np['spectra'] = np.column_stack([psd_ALL[key].amplitude for key in psd_ALL.keys()])
psd_ALL_np['mean'] = np.mean(psd_ALL_np['spectra'], axis = 1)
psd_ALL_np['median'] = np.median(psd_ALL_np['spectra'], axis = 1)
psd_ALL_np['std'] = np.std(psd_ALL_np['spectra'], axis = 1)
psd_ALL_np['sig'] = np.column_stack([signif_ALL_series[i].amplitude for i in range(0,len(signif_ALL_series))])

mean_ALL = psd_ALL_np['mean']
median_ALL = psd_ALL_np['median']
std_ALL = psd_ALL_np['std']
sig50_ALL = psd_ALL_np['sig'][:,0]
sig99_ALL = psd_ALL_np['sig'][:,2]

# Volc
psd_VOLC_np = {}
psd_VOLC_np['spectra'] = np.column_stack([psd_VOLC[key].amplitude for key in psd_VOLC.keys()])
psd_VOLC_np['mean'] = np.mean(psd_VOLC_np['spectra'], axis = 1)
psd_VOLC_np['median'] = np.median(psd_VOLC_np['spectra'], axis = 1)
psd_VOLC_np['std'] = np.std(psd_VOLC_np['spectra'], axis = 1)
psd_VOLC_np['sig'] = np.column_stack([signif_VOLC_series[i].amplitude for i in range(0,len(signif_ALL_series))])

mean_VOLC = psd_VOLC_np['mean']
median_VOLC = psd_VOLC_np['median']
std_VOLC = psd_VOLC_np['std']
sig50_VOLC = psd_VOLC_np['sig'][:,0]
sig99_VOLC = psd_VOLC_np['sig'][:,2]
# # Solar
# psd_SOLAR_np = {}
# psd_SOLAR_np['spectra'] = np.column_stack([psd_SOLAR[key].amplitude for key in psd_SOLAR.keys()])
# psd_SOLAR_np['mean'] = np.mean(psd_SOLAR_np['spectra'], axis = 1)
# psd_SOLAR_np['median'] = np.median(psd_SOLAR_np['spectra'], axis = 1)
# psd_SOLAR_np['std'] = np.std(psd_SOLAR_np['spectra'], axis = 1)
# psd_SOLAR_np['sig'] = np.column_stack([signif_SOLAR_series[i].amplitude for i in range(0,len(signif_ALL_series))])

# # Orbital
# psd_ORBITAL_np = {}
# psd_ORBITAL_np['spectra'] = np.column_stack([psd_ORBITAL[key].amplitude for key in psd_ORBITAL.keys()])
# psd_ORBITAL_np['mean'] = np.mean(psd_ORBITAL_np['spectra'], axis = 1)
# psd_ORBITAL_np['median'] = np.median(psd_ORBITAL_np['spectra'], axis = 1)
# psd_ORBITAL_np['std'] = np.std(psd_ORBITAL_np['spectra'], axis = 1)
# psd_ORBITAL_np['sig'] = np.column_stack([signif_ORBITAL_series[i].amplitude for i in range(0,len(signif_ALL_series))])


#%% plot FULL and VOLC w shading (scaled), 
# (minus 50% confidence)(divided by 99% confidence)

whole_spec = True

# pick which periods to showcase (years)
freq = psd_ALL['000'].frequency

xticks = [100,80,70,60,50,40,30]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]
# figure drawing 
fig = plt.figure(figsize=(25,10))

ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# plt.yscale('log')

plt.minorticks_off()

# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
ax.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
plt.ylim(-0.5,1.1)
if whole_spec:
    plt.ylim(-0.5,1.5)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

yticks = [50,90,95,99]
yticks2 = [0.0,0.9,0.95,1.0]
ax.set_yticks(yticks2)
yticks2_labels = [f'{str(y)}%' for y in yticks]
ax.set_yticklabels(yticks2_labels)

l1 = (mean_ALL-sig50_ALL)/(sig99_ALL-sig50_ALL)
l2 = (mean_VOLC-sig50_VOLC)/(sig99_VOLC-sig50_VOLC)
# l3 = (psd_SOLAR_np['mean']-psd_SOLAR_np['sig'][:,0])/psd_SOLAR_np['sig'][:,1]
# l4 = (psd_ORBITAL_np['mean']-psd_ORBITAL_np['sig'][:,0])/psd_ORBITAL_np['sig'][:,1]

p1 = ax.plot(freq,l1, color = 'blue', label = 'FULL FORCING', linewidth = 2.0, zorder = 10)

up_bnd = (mean_ALL + 2*std_ALL - sig50_ALL)/(sig99_ALL-sig50_ALL)
lo_bnd = (mean_ALL - 2*std_ALL - sig50_ALL)/(sig99_ALL-sig50_ALL)

f1 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, color = 'blue')

p2 = ax.plot(freq,l2, color = 'red', label = 'VOLCANIC ONLY', linewidth = 2.0, zorder = 5)

up_bnd = (mean_VOLC + 2*std_VOLC - sig50_VOLC)/(sig99_VOLC - sig50_VOLC)
lo_bnd = (mean_VOLC - 2*std_VOLC - sig50_VOLC)/(sig99_VOLC - sig50_VOLC)

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, color = 'red')

# p3 = ax.plot(freq,l3, color = 'gold', label = 'SOLAR ONLY', linewidth = 2.0, zorder = 2)
# p4 = ax.plot(freq,l4, color = 'lightskyblue', label = 'ORBITAL ONLY', linewidth = 2.0, zorder = 2)


[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5, zorder = 2) for i in [0,0.9,0.95,1]]
plt.legend()
plt.title(f'MTM_forced_nw={nw}')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/GLOB/')
name = f'psd_nw{nw}_forced_ALL_VOLC_scaled'
if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% Regular fig for comparison
whole_spec = False

# pick which periods to showcase (years)
freq = psd_ALL['000'].frequency

xticks = [100,80,70,60,50,40,30]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing 
fig = plt.figure(figsize=(25,10))

ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
plt.yscale('log')

plt.minorticks_off()


# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
ax.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)

ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

l1 = mean_ALL
l2 = mean_VOLC
# l3 = (psd_SOLAR_np['mean'])
# l4 = (psd_ORBITAL_np['mean'])

p1 = ax.plot(freq,l1, color = 'blue', label = 'ALL FORCING', linewidth = 2.0, zorder = 5)
p2 = ax.plot(freq,l2, color = 'red', label = 'VOLCANIC ONLY', linewidth = 2.0, zorder = 5)
# p3 = ax.plot(freq,l3, color = 'gold', label = 'SOLAR ONLY', linewidth = 2.0, zorder = 5)
# p4 = ax.plot(freq,l4, color = 'lightskyblue', label = 'ORBITAL ONLY', linewidth = 2.0, zorder = 5)

s50_1 = ax.plot(freq,psd_ALL_np['sig'][:,0], color = 'blue',linewidth = 1.0, zorder = 2, linestyle = '--')
s50_2 = ax.plot(freq,psd_VOLC_np['sig'][:,0], color = 'red',linewidth = 1.0, zorder = 2, linestyle = '--')
# s50_3 = ax.plot(freq,psd_SOLAR_np['sig'][:,0], color = 'gold',linewidth = 1.0, zorder = 2, linestyle = '--')
# s50_4 = ax.plot(freq,psd_ORBITAL_np['sig'][:,0], color = 'lightskyblue',linewidth = 1.0, zorder = 2, linestyle = '--')

s90_1 = ax.plot(freq,psd_ALL_np['sig'][:,1], color = 'blue', linewidth = 1.0, zorder = 2, linestyle = '-.')
s90_2 = ax.plot(freq,psd_VOLC_np['sig'][:,1], color = 'red', linewidth = 1.0, zorder = 2,linestyle = '-.')
# s90_3 = ax.plot(freq,psd_SOLAR_np['sig'][:,1], color = 'gold',linewidth = 1.0, zorder = 2,linestyle = '-.')
# s90_4 = ax.plot(freq,psd_ORBITAL_np['sig'][:,1], color = 'lightskyblue', linewidth = 1.0, zorder = 2,linestyle = '-.')

s99_1 = ax.plot(freq,psd_ALL_np['sig'][:,2], color = 'blue', linewidth = 1.0, zorder = 2, linestyle = '-.')
s99_2 = ax.plot(freq,psd_VOLC_np['sig'][:,2], color = 'red', linewidth = 1.0, zorder = 2,linestyle = '-.')
# s90_3 = ax.plot(freq,psd_SOLAR_np['sig'][:,1], color = 'gold',linewidth = 1.0, zorder = 2,linestyle = '-.')
# s90_4 = ax.plot(freq,psd_ORBITAL_np['sig'][:,1], color = 'lightskyblue', linewidth = 1.0, zorder = 2,linestyle = '-.')

plt.legend()
plt.title(f'NA_MTM_forced_nw={nw}_regular_scale')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/GLOB/')
name = f'psd_nw{nw}_forced_ALL_VOLC_reg'
if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()



#%% create pyleoclim ensemble series objects UNFORCED

# =============================================================================
# ALL FORCING ensemble
# =============================================================================

data = CESM_LME_unforced['ALL_FORCING'].sel(lat = lats).mean(dim=['lat','lon'])

series_ALL_unf = {}
for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_ALL_unf[f'{run:03}'] = series


# =============================================================================
# VOlcanic  ensemble
# =============================================================================
data = CESM_LME_unforced['VOLC'].sel(lat = lats).mean(dim=['lat','lon'])

series_VOLC_unf={}

for run, dat in data.groupby('run'):
    series = pyleo.Series(
        time=dat.year,
        value=dat.TS.to_numpy(),
        time_name="years AD",
        time_unit="yr",
        value_name="GMSAT",
        value_unit="K",
        label=run)

    series_VOLC_unf[f'{run:03}'] = series


del data, run, dat, series, 

#%% MTM  psd analysis

# bandwidth parameter
nw = 2
N = len(series_ALL_unf['000'].time)
npad = 2**int(np.ceil(np.log2(abs(N)))+2)

# All forcing
psd_ALL_unf = {}

for run, serie in series_ALL_unf.items():
    psd_ALL_unf[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})

# Volcanic
psd_VOLC_unf = {}

for run, serie in series_VOLC_unf.items():
    psd_VOLC_unf[run] = serie.detrend().spectral(method = 'mtm',
                                            settings ={
                                                'NW' : nw, 'nfft':npad})


#%%Significance tests

sig_n = 10000
qs = [0.50,0.90]

# All forcing
signif_ALL_unf = psd_ALL_unf['000'].signif_test(
    number=sig_n, qs=qs)
signif_ALL_unf_series = signif_ALL_unf.signif_qs.psd_list

# Volcanic
signif_VOLC_unf = psd_VOLC_unf['000'].signif_test(
    number=sig_n, qs=qs)
signif_VOLC_unf_series = signif_VOLC_unf.signif_qs.psd_list

#%% Get results in numpy arrays (out of Pyleoclim)

# All Forcing
psd_ALL_unf_np = {}
psd_ALL_unf_np['spectra'] = np.column_stack([psd_ALL_unf[key].amplitude for key in psd_ALL_unf.keys()])
psd_ALL_unf_np['mean'] = np.mean(psd_ALL_unf_np['spectra'], axis = 1)
psd_ALL_unf_np['median'] = np.median(psd_ALL_unf_np['spectra'], axis = 1)
psd_ALL_unf_np['std'] = np.std(psd_ALL_unf_np['spectra'], axis = 1)
psd_ALL_unf_np['sig'] = np.column_stack([signif_ALL_unf_series[i].amplitude for i in range(0,len(signif_ALL_unf_series))])

# Volc
psd_VOLC_unf_np = {}
psd_VOLC_unf_np['spectra'] = np.column_stack([psd_VOLC_unf[key].amplitude for key in psd_VOLC_unf.keys()])
psd_VOLC_unf_np['mean'] = np.mean(psd_VOLC_unf_np['spectra'], axis = 1)
psd_VOLC_unf_np['median'] = np.median(psd_VOLC_unf_np['spectra'], axis = 1)
psd_VOLC_unf_np['std'] = np.std(psd_VOLC_unf_np['spectra'], axis = 1)
psd_VOLC_unf_np['sig'] = np.column_stack([signif_VOLC_unf_series[i].amplitude for i in range(0,len(signif_VOLC_unf_series))])

#%% ALL FORCING (forced vs unforced) scaled w. confidence ints
# pick which periods to showcase (years)
freq = psd_ALL['000'].frequency
xticks = [100,80,70,60,50,40,30]#,20,10,5,3]

# figure drawing 
fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ax.set_xscale('log')
# ax.set_yscale('log')
plt.minorticks_off()

# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
ax.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
plt.ylim(-2,2)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# =============================================================================
# all forcing forced
# =============================================================================
l1 = (psd_ALL_np['mean'])
sig50 = psd_ALL_np['sig'][:,0]
sig90 = psd_ALL_np['sig'][:,1]
std = psd_ALL_np['std']
# p = [ax.plot(freq,
#               ((psd_ALL_np['spectra'][:,i]-sig50)/(sig90-sig50)),
#               linewidth = 0.5,
#               alpha = 0.5) 
#       for i in range(0,13)]
# p12 = ax.plot(freq,
#               ((sig50-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')
# p13 = ax.plot(freq,
#               ((sig90-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')

p1 = ax.plot(freq,
             ((l1-sig50)/(sig90-sig50)),
             color = 'blue', label = 'ALL FORCING', linewidth = 2.0, zorder = 5)


up_bnd = (l1+2*std-sig50)/(sig90-sig50)
lo_bnd = (l1-2*std-sig50)/(sig90-sig50)

f1 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, 
                      color = 'blue',
                      hatch = '\\')

# =============================================================================
# All forcing unforced
# =============================================================================

l1 = (psd_ALL_unf_np['mean'])
sig50 = psd_ALL_unf_np['sig'][:,0]
sig90 = psd_ALL_unf_np['sig'][:,1]
std = psd_ALL_unf_np['std']

# p = [ax.plot(freq,
#               ((psd_ALL_np['spectra'][:,i]-sig50)/(sig90-sig50)),
#               linewidth = 0.5,
#               alpha = 0.5) 
#       for i in range(0,13)]
# p12 = ax.plot(freq,
#               ((sig50-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')
# p23 = ax.plot(freq,
#               ((sig90-sig50)/(sig90-sig50)),
#               color = 'darkblue',linewidth = 1, linestyle = '--')
p2 = ax.plot(freq,
             ((l1-sig50)/(sig90-sig50)),
             color = 'darkblue', 
             label = 'ALL FORCING UNFORCED', 
             linewidth = 2.0, zorder = 5,
             linestyle = '-.')

up_bnd = (l1+2*std-sig50)/(sig90-sig50)
lo_bnd = (l1-2*std-sig50)/(sig90-sig50)

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, 
                      color = 'darkblue',
                      hatch = '//')



[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5, zorder = 2) for i in [0,1]]
plt.legend()
plt.title(f'NA_MTM_forced_nw={nw}')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/')
name = f'psd_nw{nw}_ALL_FORCING_forced_unforced_scaled_GLOBAL'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% VOLC (forced vs unforced) scaled w. confidence ints
# pick which periods to showcase (years)
freq = psd_VOLC['000'].frequency
xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing 
fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# plt.yscale('log')
plt.minorticks_off()

# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
ax.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
plt.ylim(-2,2)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# =============================================================================
# VOLC forced
# =============================================================================
l1 = (psd_VOLC_np['mean'])
sig50 = psd_VOLC_np['sig'][:,0]
sig90 = psd_VOLC_np['sig'][:,1]
std = psd_VOLC_np['std']

# p = [ax.plot(freq,
#               ((psd_ALL_np['spectra'][:,i]-sig50)/(sig90-sig50)),
#               linewidth = 0.5,
#               alpha = 0.5) 
#       for i in range(0,13)]
# p12 = ax.plot(freq,
#               ((sig50-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')
# p13 = ax.plot(freq,
#               ((sig90-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')

p1 = ax.plot(freq,
             ((l1-sig50)/(sig90-sig50)),
             color = 'red', label = 'VOLC', linewidth = 2.0, zorder = 5)


up_bnd = (l1+2*std-sig50)/(sig90-sig50)
lo_bnd = (l1-2*std-sig50)/(sig90-sig50)

f1 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, 
                      color = 'red',
                      hatch = '\\')

# =============================================================================
# VOlc unforced
# =============================================================================
l1 = (psd_VOLC_unf_np['mean'])
sig50 = psd_VOLC_unf_np['sig'][:,0]
sig90 = psd_VOLC_unf_np['sig'][:,1]
std = psd_VOLC_unf_np['std']

# p = [ax.plot(freq,
#               ((psd_ALL_np['spectra'][:,i]-sig50)/(sig90-sig50)),
#               linewidth = 0.5,
#               alpha = 0.5) 
#       for i in range(0,13)]
# p12 = ax.plot(freq,
#               ((sig50-sig50)/(sig90-sig50)),
#               color = 'blue',linewidth = 1, linestyle = '--')
# p23 = ax.plot(freq,
#               ((sig90-sig50)/(sig90-sig50)),
#               color = 'darkblue',linewidth = 1, linestyle = '--')
p2 = ax.plot(freq,
             ((l1-sig50)/(sig90-sig50)),
             color = 'darkred', 
             label = 'VOLC UNFORCED', 
             linewidth = 2.0, zorder = 5,
             linestyle = '-.')

up_bnd = (l1+2*std-sig50)/(sig90-sig50)
lo_bnd = (l1-2*std-sig50)/(sig90-sig50)

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 1, 
                      color = 'darkred',
                      hatch = '//')


[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5, zorder = 2) for i in [0,1]]
plt.legend()
plt.title(f'NA_MTM_forced_nw={nw}')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/')
name = f'psd_nw{nw}_VOLC_forced_unforced_scaled_GLOBAL_whole_spectrum'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
