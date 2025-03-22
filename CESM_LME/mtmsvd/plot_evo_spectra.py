#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:08:49 2024

@author: afer
"""

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv_evo/')

files = [entry for entry in os.listdir(path) if not entry.startswith('.')]

cmap = 'inferno'

# load all evo spectra
ix = next((i for i, file in enumerate(files) if '_forc.nc' in file), None)
lfv = xr.open_mfdataset(path+files[ix])

ix = next((i for i, file in enumerate(files) if '_unforced.nc' in file), None)
lfv_unforced = xr.open_mfdataset(path+files[ix])

ix = next((i for i, file in enumerate(files) if '_forc_NA.nc' in file), None)
lfv_NA = xr.open_mfdataset(path+files[ix])

ix = next((i for i, file in enumerate(files) if '_unforced_NA.nc' in file), None)
lfv_unforced_NA = xr.open_mfdataset(path+files[ix])

ix = next((i for i, file in enumerate(files) if '_EM.nc' in file), None)
lfv_EM = xr.open_mfdataset(path+files[ix])

# =============================================================================
# load confidence intervals
# =============================================================================

ci_file = [entry for entry in files if not entry.startswith('.') and entry.startswith('conf_int')]

ci = {}
for files in ci_file:
    ci[files[9:-4]] = np.load(path+files)

ci_NA = {}
for cii in ci.items():
    if 'NA' in cii[0]:
        ci_NA[cii[0]] = cii[1]

ci_GLOB = {}
for cii in ci.items():
    if 'NA' not in cii[0]:
        ci_GLOB[cii[0]] = cii[1]

# =============================================================================
# Determine significance intervals to plot contours
# =============================================================================

ci50_ii = (4,1)
ci90_ii = (2,1)
ci99_ii = (0,1)

ci50_GLOB = []
ci90_GLOB = []
ci99_GLOB = []

ci50_NA = []
ci90_NA = []
ci99_NA = []

for cis in ci_GLOB.values():
    ci50_GLOB.append(cis[ci50_ii])
    ci90_GLOB.append(cis[ci90_ii])
    ci99_GLOB.append(cis[ci99_ii])
for cis in ci_NA.values():
    ci50_NA.append(cis[ci50_ii])
    ci90_NA.append(cis[ci90_ii])
    ci99_NA.append(cis[ci99_ii])
ci50_GLOB = np.mean(ci50_GLOB)
ci90_GLOB = np.mean(ci90_GLOB)
ci99_GLOB = np.mean(ci99_GLOB)
ci50_NA = np.mean(ci50_NA)
ci90_NA = np.mean(ci90_NA)
ci99_NA = np.mean(ci99_NA)
    
del files, ci_file
#%%

# Organize data into cases
lfv_dic = {}
lfv_unforced_dic = {}
lfv_NA_dic = {}
lfv_unforced_NA_dic = {}
lfv_EM_dic = {}

# put lfv data into dictionary for ease of handling
for key, value in lfv.items():
    lfv_dic[key]=value
lfv = lfv_dic

for key, value in lfv_unforced.items():
    lfv_unforced_dic[key]=value
lfv_unf = lfv_unforced_dic

for key, value in lfv_NA.items():
    lfv_NA_dic[key]=value
lfv_NA = lfv_NA_dic

for key, value in lfv_unforced_NA.items():
    lfv_unforced_NA_dic[key]=value
lfv_unf_NA = lfv_unforced_NA_dic

for key, value in lfv_EM.items():
    lfv_EM_dic[key]=value
lfv_EM = lfv_EM_dic

# =============================================================================
# # create lfv dictionary by case 
# =============================================================================

#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case = {}

for case_i in cases:
    case_str = [key for key in lfv.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case[case_i] = ds



#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv_unf.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case_unf = {}

for case_i in cases:
    case_str = [key for key in lfv_unf.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv_unf[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case_unf[case_i] = ds



#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv_NA.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case_NA = {}

for case_i in cases:
    case_str = [key for key in lfv_NA.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv_NA[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case_NA[case_i] = ds



#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv_unf_NA.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case_unf_NA = {}

for case_i in cases:
    case_str = [key for key in lfv_unf_NA.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv_unf_NA[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case_unf_NA[case_i] = ds


#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv_EM.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case_EM = {}

for case_i in cases:
    case_str = [key for key in lfv_EM.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv_EM[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case_EM[case_i] = ds


del ds, case_i, key, value, case_str, lfv_dic, lfv_unforced_dic
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

#%% Cycle through all runs of a case and plot Evo LFV spectra

case = 'ALL'
NA = True
whole_spec = False


save_fig = input("Save fig? (y/n):").lower()


for run_i in range(len(lfv_by_case[case].run)):
    
    fig = plt.figure(figsize=(25,10))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax1.minorticks_off()
    ax2.minorticks_off()
    
    # pick which periods to showcase (years)
    yticks = [100,80,70,60,50,40,30,20]
    if whole_spec:
        yticks = [100,80,70,60,50,40,30,20,10,5,3]
    
    yticks2 = [1/x for x in yticks]
    ax1.set_yticks(yticks2)
    ax2.set_yticks(yticks2)
    yticks2_labels = [str(x) for x in yticks]
    ax1.set_yticklabels(yticks2_labels)
    ax2.set_yticklabels(yticks2_labels)
    ax1.tick_params(axis='x',which='major',direction='out',
                   pad=15, labelrotation=45)
    ax2.tick_params(axis='x',which='major',direction='out',
                   pad=15, labelrotation=45)
    ax1.set_xlabel('year')
    ax2.set_xlabel('year')
    ax1.set_ylabel('Period [yrs]')
    ax2.set_ylabel('Period [yrs]')
    
    #confidence intervals
    ci50 = ci50_GLOB if not NA else ci50_NA
    ci90 = ci90_GLOB if not NA else ci90_NA
    ci99 = ci99_GLOB if not NA else ci99_NA


    # data to plot
    dat = lfv_by_case
    if NA:
        dat = lfv_by_case_NA
    dat_unf = lfv_by_case_unf
    if NA:
        dat_unf = lfv_by_case_unf_NA

    p1 = dat[case].sel(run = run_i).plot.contourf(
        ax = ax1,
        add_labels = False,
        x='mid_yr',
        y='freq',
        ylim = [1/yticks[0],1/yticks[-1]],
        # robust = True,
        vmin = 0.4,
        vmax = 0.8,# if NA else 0.8,
        cmap = cmap,
        levels = 50
        )
    p1_ci = dat[case].sel(run = run_i).plot.contour(
        ax = ax1,
        add_labels = False,
        x='mid_yr',
        y='freq',
        ylim = [1/yticks[0],1/yticks[-1]],
        levels = [ci50,ci90,ci99],
        colors = ['black','black','black'],
        linewidths = [1,2,2.5],
        linestyles = ['dashed','solid','solid']
        )
    
    p2 = dat_unf[case].sel(run = run_i).plot.contourf(
        ax = ax2,
        add_labels = False,
        x='mid_yr',
        y='freq',
        ylim = [1/yticks[0],1/yticks[-1]],
        # robust = True,
        vmin = 0.4,
        vmax = 0.8,# if NA else 0.8,
        cmap = cmap,
        levels = 50
        )
    p2_ci = dat_unf[case].sel(run = run_i).plot.contour(
        ax = ax2,
        add_labels = False,
        x='mid_yr',
        y='freq',
        ylim = [1/yticks[0],1/yticks[-1]],
        levels = [ci50,ci90,ci99],
        colors = ['black','black','black'],
        linewidths = [1,2,2.5],
        linestyles = ['dashed','solid','solid']
        )

    ax1.set_title(f'{case} run {run_i}')
    ax2.set_title(f'{case} run {run_i} unforced')
    if NA:
        ax1.set_title(f'{case} run {run_i} NA')
        ax2.set_title(f'{case} run {run_i} unforced NA')

    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/GLOB/')
    if NA:
        save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/NA/')
    
    name = f'lfv_evo_{case}_run{run_i}'
    name = name + '_NA' if NA else name
    
    if whole_spec:
        name = name + '_whole_spec'
    
    
    if save_fig == 'y':
        plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
        plt.savefig(save_path+name+'.svg', format = 'svg')
    
    else:
        plt.show()
        

#%% Plot mean of a case
case = 'VOLC'
NA = False
whole_spec = False
dat = lfv_by_case_NA if NA else lfv_by_case
dat_unf = lfv_by_case_unf_NA if NA else lfv_by_case_unf

fig = plt.figure(figsize=(25,10))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.set_yscale('log')
ax2.set_yscale('log')

ax1.minorticks_off()
ax2.minorticks_off()

# pick which periods to showcase (years)
yticks = [100,90,80,70,60,50,40,30,20]
if whole_spec:
    yticks = [100,80,70,60,50,40,30,20,10,5,3]

yticks2 = [1/x for x in yticks]
ax1.set_yticks(yticks2)
ax2.set_yticks(yticks2)
yticks2_labels = [str(x) for x in yticks]
ax1.set_yticklabels(yticks2_labels)
ax2.set_yticklabels(yticks2_labels)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax2.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.set_xlabel('year')
ax2.set_xlabel('year')
ax1.set_ylabel('Period [yrs]')
ax2.set_ylabel('Period [yrs]')

#confidence intervals
ci50 = ci50_GLOB if not NA else ci50_NA
ci90 = ci90_GLOB if not NA else ci90_NA
ci99 = ci99_GLOB if not NA else ci99_NA

# ax1.axhline(y=1/95, color='black', linestyle='-')

dat[case].mean(dim = 'run').plot.contourf(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    # robust = True,
    vmin = 0.4,
    vmax = 0.8 if NA else 0.8,
    cmap = cmap,
    levels = 15
    )

dat[case].mean(dim = 'run').plot.contour(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    levels = [ci50,ci90,ci99],
    colors = ['black','black','black'],
    linewidths = [1,2,2.5],
    linestyles = ['dashed','solid','solid']
    )

dat_unf[case].mean(dim = 'run').plot.contourf(
    ax = ax2,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    # robust = True,
    vmin = 0.4,
    vmax = 0.8 if NA else 0.8,
    cmap = cmap,
    levels = 15
    )
dat_unf[case].mean(dim = 'run').plot.contour(
    ax = ax2,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    levels = [ci50,ci90,ci99],
    colors = ['black','black','black'],
    linewidths = [1,2,2.5],
    linestyles = ['dashed','solid','solid']
    )
title1 = f'{case} mean forced NA' if NA else f'{case} mean forced'
ax1.set_title(title1)

title2 = f'{case} mean unforced NA' if NA else f'{case} mean unforced'
ax2.set_title(title2)
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/GLOB/')
if NA:
    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/NA/')

name = f'lfv_evo_{case}_mean'

if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
#%% Plot ensemble mean spectra

case = 'ALL'
NA = True
whole_spec = False

dat =  lfv_by_case_EM

fig = plt.figure(figsize=(25,10))

ax1 = plt.subplot(121)

ax1.set_yscale('log')

ax1.minorticks_off()

# pick which periods to showcase (years)
yticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    yticks = [100,80,70,60,50,40,30,20,10,5,3]

yticks2 = [1/x for x in yticks]
ax1.set_yticks(yticks2)
yticks2_labels = [str(x) for x in yticks]
ax1.set_yticklabels(yticks2_labels)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.set_xlabel('year')
ax1.set_ylabel('Period [yrs]')

#confidence intervals
ci50 = ci50_GLOB if not NA else ci50_NA
ci90 = ci90_GLOB if not NA else ci90_NA
ci99 = ci99_GLOB if not NA else ci99_NA


dat[case].sel(run = 0).plot.contourf(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    # robust = True,
    vmin = 0.4,
    # vmax = 0.8 if NA else 0.8,
    robust= True,
    cmap = cmap,
    levels = 50
    )

dat[case].sel(run = 0).plot.contour(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    levels = [ci50,ci90,ci99],
    colors = ['black','black','black'],
    linewidths = [1,2,2.5],
    linestyles = ['dashed','solid','solid']
    )

title1 = f'{case} mean forced NA' if NA else f'{case} mean forced'
ax1.set_title(title1)


save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/GLOB/')
if NA:
    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/NA/')

name = f'lfv_evo_{case}_mean'

if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()


#%% Plot CNTL

NA = True
whole_spec = False

dat =  lfv_by_case if not NA else lfv_by_case_NA

fig = plt.figure(figsize=(25,10))

ax1 = plt.subplot(121)

ax1.set_yscale('log')

ax1.minorticks_off()

# pick which periods to showcase (years)
yticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    yticks = [100,80,70,60,50,40,30,20,10,5,3]

yticks2 = [1/x for x in yticks]
ax1.set_yticks(yticks2)
yticks2_labels = [str(x) for x in yticks]
ax1.set_yticklabels(yticks2_labels)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.set_xlabel('year')
ax1.set_ylabel('Period [yrs]')

#confidence intervals
ci50 = ci50_GLOB if not NA else ci50_NA
ci90 = ci90_GLOB if not NA else ci90_NA
ci99 = ci99_GLOB if not NA else ci99_NA


dat['CNTL'].sel(run = 0).plot.contourf(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    # robust = True,
    vmin = 0.4,
    # vmax = 0.8 if NA else 0.8,
    robust= True,
    cmap = cmap,
    levels = 50
    )

dat['CNTL'].sel(run = 0).plot.contour(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    levels = [ci50,ci90,ci99],
    colors = ['black','black','black'],
    linewidths = [1,2,2.5],
    linestyles = ['dashed','solid','solid']
    )

title1 = f'CNTL NA' if NA else f'CNTL'
ax1.set_title(title1)


save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/GLOB/')
if NA:
    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv_evo/NA/')

name = f'lfv_evo_{case}_mean'

if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
