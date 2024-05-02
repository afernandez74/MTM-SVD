import pyleoclim as pyleo
import xarray as xr
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import numpy as np
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
data = CESM_LME['ALL_FORCING'].mean(dim=['lat','lon'])

series_ALL = {}
series_ens_ALL=[]
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
    series_ens_ALL.append(series)

series_ens_ALL = pyleo.EnsembleSeries(series_ens_ALL)

# =============================================================================
# VOlcanic  ensemble
# =============================================================================
data = CESM_LME['VOLC'].mean(dim=['lat','lon'])

series_VOLC={}
series_ens_VOLC=[]

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
    series_ens_VOLC.append(series)

series_ens_VOLC = pyleo.EnsembleSeries(series_ens_VOLC)

del data, run, dat, series, 
#%% some plots
filt_t = 30

# stack plots
# fig, ax = series_ens_ALL.standardize().stackplot(figsize = (10,30))
# fig, ax = series_ens_VOLC.stackplot(figsize = (10,10))

# envelope
series_ens_ALL.standardize().slice(timespan=[850+filt_t,1850-filt_t]).detrend(
    method = 'linear').filter(
        cutoff_freq = 1/filt_t,method = 'butterworth').plot_envelope(
            qs = [0.005, 0.25, 0.5, 0.75, 0.995], curve_lw=1,
            shade_clr = 'teal', curve_clr='darkblue', 
            lgd_kwargs={'ncol':3,'loc':'upper center'},
            outer_shade_label='99% CI', title='ALL FORCING GLOBAL ensemble spread',
            figsize = (30,10)) 

    
# save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
# name = f'gmsat_{filt_t}_lowpass_LM_forced'

# save_fig = input("Save fig? (y/n):").lower()

# if save_fig == 'y':
#     plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
#     plt.savefig(save_path+name+'.svg', format = 'svg')

# else:
#     plt.show()
#%%

nw = 2
psd_ALL_mtm = {}

for run, serie in series_ALL.items():
    psd_ALL_mtm[run] = serie.detrend().spectral(method = 'mtm', settings ={
        'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_VOLC_mtm = {}

for run, serie in series_VOLC.items():
    psd_VOLC_mtm[run] = serie.detrend().spectral(method = 'mtm', settings ={
        'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_ALL_ens_mtm = series_ens_ALL.spectral(method = 'mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_VOLC_ens_mtm = series_ens_VOLC.spectral(method = 'mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False})

signif_ALL = series_ALL['000'].detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False}).signif_test(
    number=10000, qs=[0.50,0.90,0.99])
signif_ALL_series = signif_ALL.signif_qs.psd_list
        
signif_VOLC = series_VOLC['000'].detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False}).signif_test(
    number=10000, qs=[0.50,0.90,0.99])
signif_VOLC_series = signif_VOLC.signif_qs.psd_list


#%% 

psd_ALL_mtm_arr = np.column_stack([psd_ALL_mtm[key].amplitude for key in psd_ALL_mtm.keys()])

psd_ALL_mtm_mean = np.mean(psd_ALL_mtm_arr, axis = 1)
psd_ALL_mtm_median = np.median(psd_ALL_mtm_arr, axis = 1)
psd_ALL_mtm_std = np.std(psd_ALL_mtm_arr, axis = 1)

signif_ALL_arr = np.column_stack([signif_ALL_series[i].amplitude for i in range(0,len(signif_ALL_series))])

psd_VOLC_mtm_arr = np.column_stack([psd_VOLC_mtm[key].amplitude for key in psd_VOLC_mtm.keys()])

psd_VOLC_mtm_mean = np.mean(psd_VOLC_mtm_arr, axis = 1)
psd_VOLC_mtm_median = np.median(psd_VOLC_mtm_arr, axis = 1)
psd_VOLC_mtm_std = np.std(psd_VOLC_mtm_arr, axis = 1)

signif_VOLC_arr = np.column_stack([signif_VOLC_series[i].amplitude for i in range(0,len(signif_VOLC_series))])

#%% plot ensemble psd

fig, ax = signif_ALL.plot(figsize = (30,10),signif_linewidth=2, signif_clr='black',linewidth=0.0,
                          title = f'mtm analysis nw={nw} ALL FORCING GLOBAL FORCED')
psd_ALL_ens_mtm.plot_envelope(ax = ax, in_period = True, curve_clr='blue', shade_clr='blue')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/')
name = f'psd_mtm_nw{nw}_forced_ALL_GLOBAL'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

fig, ax = signif_VOLC.plot(figsize = (30,10),signif_linewidth=2, signif_clr='black',linewidth=0.0,
                           title = f'mtm analysis nw={nw} VOLC GLOBAL FORCED')
psd_VOLC_ens_mtm.plot_envelope(ax = ax, in_period = True, curve_clr='red', shade_clr='red')

name = f'psd_mtm_nw{nw}_forced_VOLC_GLOB'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% separate y axis for ALL and VOLC

# pick which periods to showcase (years)
freq = psd_ALL_mtm['000'].frequency
xticks = [100,80,70,60,50,40,30]

# figure drawing

fig,ax1 = plt.subplots(figsize=(20, 10))
plt.xscale('log')
plt.minorticks_off()

# plt.yscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax1.set_xticklabels(xticks2_labels)
ax1.grid(True,which='major',axis='both')
ax1.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
# plt.ylim(0.0,0.6)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
# =============================================================================
# plot ALL line
# =============================================================================
ax1.set_ylim(-0.0,1)
color = 'blue'
ax1.set_ylabel('PSD ALL FORCING', color=color)
ax1.tick_params(axis = 'y',labelcolor = color)

# plot significance
linestyles = ['dashed','dashdot','dotted']
[ax1.plot(freq,signif_ALL_arr[:,i], linewidth = 3.0, color ='blue', linestyle = linestyles[i]) for i in range(0,2)] #ALL
#CHANGE RANGE to 0,3 for 99% conf

p1 = ax1.plot(freq,psd_ALL_mtm_mean, color = 'blue', label = 'ALL FORCING', linewidth = 3.0)
# [plt.plot(freq,psd_ALL_mtm_arr[:,i], linewidth = 0.5, alpha = 0.5) for i in range(0,12)]

up_bnd = psd_ALL_mtm_mean + psd_ALL_mtm_std
lo_bnd = psd_ALL_mtm_mean - psd_ALL_mtm_std

f1 = ax1.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'blue')

# =============================================================================
# plot VOLC lines
# =============================================================================
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('PSD VOLC ONLY', color=color)
ax2.tick_params(axis = 'y',labelcolor = color)

ax2.set_ylim(0,1)
[ax2.plot(freq,signif_VOLC_arr[:,i], linewidth = 3.0, color ='red',linestyle = linestyles[i]) for i in range(0,2)] #VOLC

p2 = ax2.plot(freq,psd_VOLC_mtm_mean,color='red', label = 'VOLCANIC ONLY', linewidth = 3.0)
# [plt.plot(freq,psd_VOLC_mtm_arr[:,i], linewidth = 0.5, alpha = 0.5) for i in range(0,12)]

up_bnd = psd_VOLC_mtm_mean + psd_VOLC_mtm_std
lo_bnd = psd_VOLC_mtm_mean - psd_VOLC_mtm_std

f2 = ax2.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'red')
fig.tight_layout() 
plt.legend()
plt.title('GLOBAL surface temperature MTM PSD')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/')
name = f'psd_mtm_nw{nw}_forced_ALL_VOLC_2axis_GLOB'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% create pyleoclim ensemble series objects 
# =============================================================================
# ALL FORCING ensemble
# =============================================================================
data = CESM_LME_unforced['ALL_FORCING'].mean(dim=['lat','lon'])

series_ALL = {}
series_ens_ALL=[]
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
    series_ens_ALL.append(series)

series_ens_ALL = pyleo.EnsembleSeries(series_ens_ALL)

# =============================================================================
# VOlcanic  ensemble
# =============================================================================
data = CESM_LME_unforced['VOLC'].mean(dim=['lat','lon'])

series_VOLC={}
series_ens_VOLC=[]

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
    series_ens_VOLC.append(series)

series_ens_VOLC = pyleo.EnsembleSeries(series_ens_VOLC)

del data, run, dat, series, 
#%% some plots
filt_t = 30

# stack plots
# fig, ax = series_ens_ALL.standardize().stackplot(figsize = (10,30))
# fig, ax = series_ens_VOLC.stackplot(figsize = (10,10))

# envelope
series_ens_ALL.standardize().slice(timespan=[850+filt_t,1850-filt_t]).detrend(
    method = 'linear').filter(
        cutoff_freq = 1/filt_t,method = 'butterworth').plot_envelope(
            qs = [0.005, 0.25, 0.5, 0.75, 0.995], curve_lw=1,
            shade_clr = 'teal', curve_clr='darkblue', 
            lgd_kwargs={'ncol':3,'loc':'upper center'},
            outer_shade_label='99% CI', title='ALL FORCING GLOBAL ensemble spread',
            figsize = (30,10)) 

    
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/data/')
name = f'gmsat_{filt_t}_lowpass_LM'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()


#%% Spectral analysis (varying nw parameter)
nw = 2
psd_ALL_mtm = {}

for run, serie in series_ALL.items():
    psd_ALL_mtm[run] = serie.detrend().spectral(method = 'mtm', settings ={
        'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_VOLC_mtm = {}

for run, serie in series_VOLC.items():
    psd_VOLC_mtm[run] = serie.detrend().spectral(method = 'mtm', settings ={
        'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_ALL_ens_mtm = series_ens_ALL.spectral(method = 'mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False})

psd_VOLC_ens_mtm = series_ens_VOLC.spectral(method = 'mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False})

signif_ALL = series_ALL['000'].detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False}).signif_test(
    number=10000, qs=[0.50,0.90,0.99])
signif_ALL_series = signif_ALL.signif_qs.psd_list
        
signif_VOLC = series_VOLC['000'].detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'standardize' : False, 'adaptive' : False}).signif_test(
    number=10000, qs=[0.50,0.90,0.99])
signif_VOLC_series = signif_VOLC.signif_qs.psd_list


#%% better plot psd (not ensemble Pyleoclim series bs)

psd_ALL_mtm_arr = np.column_stack([psd_ALL_mtm[key].amplitude for key in psd_ALL_mtm.keys()])

psd_ALL_mtm_mean = np.mean(psd_ALL_mtm_arr, axis = 1)
psd_ALL_mtm_median = np.median(psd_ALL_mtm_arr, axis = 1)
psd_ALL_mtm_std = np.std(psd_ALL_mtm_arr, axis = 1)

signif_ALL_arr = np.column_stack([signif_ALL_series[i].amplitude for i in range(0,len(signif_ALL_series))])

psd_VOLC_mtm_arr = np.column_stack([psd_VOLC_mtm[key].amplitude for key in psd_VOLC_mtm.keys()])

psd_VOLC_mtm_mean = np.mean(psd_VOLC_mtm_arr, axis = 1)
psd_VOLC_mtm_median = np.median(psd_VOLC_mtm_arr, axis = 1)
psd_VOLC_mtm_std = np.std(psd_VOLC_mtm_arr, axis = 1)

signif_VOLC_arr = np.column_stack([signif_VOLC_series[i].amplitude for i in range(0,len(signif_VOLC_series))])

#%%separate y axis for ALL and VOLC

# pick which periods to showcase (years)
freq = psd_ALL_mtm['000'].frequency
xticks = [100,80,70,60,50,40,30]

# figure drawing

fig,ax1 = plt.subplots(figsize=(20, 10))
plt.xscale('log')
plt.minorticks_off()

# plt.yscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax1.set_xticklabels(xticks2_labels)
ax1.grid(True,which='major',axis='both')
ax1.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
# plt.ylim(0.0,0.6)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
# =============================================================================
# plot ALL line
# =============================================================================
ax1.set_ylim(-0.0,0.4)
color = 'blue'
ax1.set_ylabel('PSD ALL FORCING', color=color)
ax1.tick_params(axis = 'y',labelcolor = color)

# plot significance
linestyles = ['dashed','dashdot','dotted']
[ax1.plot(freq,signif_ALL_arr[:,i], linewidth = 3.0, color ='blue', linestyle = linestyles[i]) for i in range(0,2)] #ALL
#CHANGE RANGE to 0,3 for 99% conf

p1 = ax1.plot(freq,psd_ALL_mtm_mean, color = 'blue', label = 'ALL FORCING', linewidth = 3.0)
# [plt.plot(freq,psd_ALL_mtm_arr[:,i], linewidth = 0.5, alpha = 0.5) for i in range(0,12)]

up_bnd = psd_ALL_mtm_mean + psd_ALL_mtm_std
lo_bnd = psd_ALL_mtm_mean - psd_ALL_mtm_std

f1 = ax1.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'blue')

# =============================================================================
# plot VOLC lines
# =============================================================================
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('PSD VOLC ONLY', color=color)
ax2.tick_params(axis = 'y',labelcolor = color)

ax2.set_ylim(0,0.39)
[ax2.plot(freq,signif_VOLC_arr[:,i], linewidth = 3.0, color ='red',linestyle = linestyles[i]) for i in range(0,2)] #VOLC

p2 = ax2.plot(freq,psd_VOLC_mtm_mean,color='red', label = 'VOLCANIC ONLY', linewidth = 3.0)
# [plt.plot(freq,psd_VOLC_mtm_arr[:,i], linewidth = 0.5, alpha = 0.5) for i in range(0,12)]

up_bnd = psd_VOLC_mtm_mean + psd_VOLC_mtm_std
lo_bnd = psd_VOLC_mtm_mean - psd_VOLC_mtm_std

f2 = ax2.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'red')
fig.tight_layout() 
plt.legend()
plt.title('GLOBAL surface temperature MTM PSD unforced')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/psd/')
name = f'psd_mtm_nw{nw}_unforced_ALL_VOLC_2axis_GLOB'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()