#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:03:27 2024

@author: afer
"""


# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% load data

path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')
files = os.listdir(path)

# =============================================================================
# file names
# =============================================================================
lfv_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('lfv_NA.nc')]

lfv_unforced_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('lfv_unforced_NA')]

lfv_obs_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('HadCRUT5')]

ci_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int_VOLC_run1_NA.npy')]

ci_obs_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int_HadCRUT')]

# =============================================================================
# load .nc files with xr
# =============================================================================
lfv = xr.open_dataset(path+lfv_file[0])
lfv_unforced = xr.open_dataset(path+lfv_unforced_file[0])
lfv_obs = xr.open_dataset(path+lfv_obs_file[0])['HadCRUT5_lfv']
ci = np.load(path+ci_file[0])
ci_obs = np.load(path+ci_obs_file[0])

del path, lfv_file, ci_file, lfv_unforced_file
#%% Organize data to plot

# Organize data into cases
lfv_dic = {}
lfv_unforced_dic = {}

# put lfv data into dictionary for ease of handling
for key, value in lfv.items():
    lfv_dic[key]=value
lfv = lfv_dic

for key, value in lfv_unforced.items():
    lfv_unforced_dic[key]=value
lfv_unf = lfv_unforced_dic

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

# create lfv dictionary by case  (for unforced data)

#string list of unique cases
cases = set(entry.split('_')[0] for entry in lfv_unf.keys())

#dictionary of lfv data split into each case, concatenated through 'run' dimension
lfv_by_case_unf = {}

for case_i in cases:
    case_str = [key for key in lfv_unf.keys() if key.startswith(case_i)]
    ds = xr.concat([lfv_unf[key] for key in case_str], dim = 'run').rename(case_i)
    lfv_by_case_unf[case_i] = ds

del ds, case_i, key, value, case_str, lfv_dic, lfv_unforced_dic

#%% adjustments
freq = lfv['ALL_FORCING_001_lfv'].freq.data
freq_obs = lfv_obs.freq

# calculate secular means for all spectra
nw = 2
N_obs = 174
N_mod = 1000
fr_sec_obs = nw/N_obs
fr_sec_mod = nw/N_mod
fr_sec_obs_ix = np.where(freq_obs < fr_sec_obs)[0][-1] 
fr_sec_mod_ix = np.where(freq < fr_sec_mod)[0][-1] 

# means
mod_mean = lfv_by_case['ALL'].sel(run=0)[fr_sec_mod_ix:].mean()
obs_mean = lfv_obs[fr_sec_obs_ix:].mean()

#medians
mod_median = lfv_by_case['ALL'].sel(run=0)[fr_sec_mod_ix:].median()
obs_median = lfv_obs[fr_sec_obs_ix:].median()


# adjust confidence intervals 
# 50% non-secular CI must match median of spectra


#%% Calculate means and stds for each case

lfv_means = {}

for key, value in lfv_by_case.items():

    if key == 'ALL': # 'ALL_FORCING' contains ensemble mean, which we don't want in the mean calculation
        lfv_means[key] = value.isel(run = slice(0,-1)).mean(dim='run')

    else:
        lfv_means[key] = value.mean(dim='run')

lfv_std = {}

for key, value in lfv_by_case.items():

    if key == 'ALL':
        lfv_std[key] = value.isel(run = slice(0,-1)).std(dim='run')

    else:
        lfv_std[key] = value.std(dim='run')

del key, value

#%% Calculate means and stds for unforced lfvs

lfv_means_unf = {}

for key, value in lfv_by_case_unf.items():

    if key == 'ALL': # 'ALL_FORCING' contains ensemble mean, which we don't want in the mean calculation
        lfv_means_unf[key] = value.isel(run = slice(0,-1)).mean(dim='run')

    else:
        lfv_means_unf[key] = value.mean(dim='run')

lfv_std_unf = {}

for key, value in lfv_by_case_unf.items():

    if key == 'ALL':
        lfv_std_unf[key] = value.isel(run = slice(0,-1)).std(dim='run')

    else:
        lfv_std_unf[key] = value.std(dim='run')

del key, value

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

#%% plot all LFV spectra means

# pick which periods to showcase (years)
whole_spec = False
xticks = [100,80,70,60,50,40,30]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(25,10))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# plt.yscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
plt.minorticks_off()
# plot lines

# case = 'CNTL'

# p1 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'darkgrey',
#         label = 'CONTROL')

case = 'ALL'

p2 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = 'ALL FORCING')


case = 'VOLC'

p3 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'red',
        label = 'VOLCANIC ONLY')


# case = 'SOLAR'

# p4 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'gold',
#         label = 'SOLAR ONLY')


# case = 'ORBITAL'

# p5 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'lightskyblue',
#         label = 'ORBITAL ONLY')

# case = 'obs'

# adjust obs spectrum to have same mean as nonsecular lfvs
# adj_obs = ci[-1,-1]/obs_mean
# p6 = (lfv_obs[fr_sec_obs_ix:]*adj_obs).plot(
#         linestyle = ':',
#         linewidth=2,
#         zorder = 5,
#         color = 'black',
#         label = 'HadCRUT5 obs')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title('NA MTM-SVD CESM LME')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/NA/')
name = 'lfv_forced_ALL_VOLC'

if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
#%% plot spectra with shaded spread

freq = lfv['CNTL_lfv'].freq.data

# pick which periods to showcase (years)

# pick which periods to showcase (years)
whole_spec = False
xticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.3,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
plt.minorticks_off()
# plot lines

# case = 'CNTL'

# p1 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 5,
#         color = 'black',
#         label = 'CONTROL')

case = 'ALL'

p2 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = 'ALL FORCING')

up_bnd = lfv_means[case].data + 2*lfv_std[case].data
lo_bnd = lfv_means[case].data - 2*lfv_std[case].data

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'blue')

case = 'VOLC'

p3 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'red',
        label = 'VOLCANIC ONLY')

up_bnd = lfv_means[case].data + 2*lfv_std[case].data
lo_bnd = lfv_means[case].data - 2*lfv_std[case].data

f3 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'red')

# case = 'SOLAR'

# p4 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'orange',
#         label = 'SOLAR ONLY')

# up_bnd = lfv_means[case].data + lfv_std[case].data
# lo_bnd = lfv_means[case].data - lfv_std[case].data

# f4 = ax.fill_between(freq, up_bnd, 
#                       lo_bnd,
#                       alpha=.1, linewidth=0, zorder = 2, color = 'orange')

# case = 'ORBITAL'

# p5 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'purple',
#         label = 'ORBITAL ONLY')

# up_bnd = lfv_means[case].data + lfv_std[case].data
# lo_bnd = lfv_means[case].data - lfv_std[case].data

# f5 = ax.fill_between(freq, up_bnd, 
#                       lo_bnd,
#                       alpha=.1, linewidth=0, zorder = 2, color = 'purple')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.5, linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title('NA MTM-SVD CESM LME')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/NA/')
name = 'lfv_forced_ALL_VOLC'
if whole_spec:
    name = name + '_whole_spec'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    

#%% Compare forced and unforced with shading

# pick which periods to showcase (years)

whole_spec = False
xticks = [100,80,70,60,50,40,30,20]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.3,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
plt.minorticks_off()

# plot lines

case = 'VOLC'
if case=='VOLC':
    color1 = 'red'
    color2 = 'darkred'

elif case=='ALL':
    color1='blue'
    color2='darkblue'

p2 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = color1,
        label = f'{case} forced')

up_bnd = lfv_means[case].data + 2*lfv_std[case].data
lo_bnd = lfv_means[case].data - 2*lfv_std[case].data

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, 
                      color = color1,
                      hatch = '\\')


p3 = lfv_means_unf[case].plot(
        linestyle = '-.',
        linewidth=2,
        zorder = 10,
        color = color2,
        label = f'{case} unforced')

up_bnd = lfv_means_unf[case].data + 2*lfv_std_unf[case].data
lo_bnd = lfv_means_unf[case].data - 2*lfv_std_unf[case].data

f3 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, 
                      color = color2,
                      hatch = '//')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]

plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME {case}')

ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/NA/')
name = f'lfv_forced_unforced_{case}'

if whole_spec:
    name = name + '_whole_spec'
    
save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    
    
# #%% Compare unforced ALL vs unforced VOLC

# # pick which periods to showcase (years)

# xticks = [100,80,70,60,50,40,30]
# # xticks = [20,10,7,5,4]

# # figure drawing

# fig = plt.figure(figsize=(30,12))
# ax = fig.add_axes([0.1,0.1,0.5,0.8])
# plt.xscale('log')

# # set x ticks and labels

# xticks2 = [1/x for x in xticks]
# ax.set_xticks(xticks2)
# xticks2_labels = [str(x) for x in xticks]
# ax.set_xticklabels(xticks2_labels)
# ax.grid(True,which='major',axis='both')
# plt.xlim((xticks2[0],xticks2[-1]))
# plt.ylim(0.4,0.8)
# ax.tick_params(axis='x',which='major',direction='out',
#                pad=15, labelrotation=45)
# plt.minorticks_off()

# # plot lines

# case = 'ALL'

# p2 = lfv_means_unf[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'blue',
#         label = f'{case} unforced')

# up_bnd = lfv_means_unf[case].data + lfv_std_unf[case].data
# lo_bnd = lfv_means_unf[case].data - lfv_std_unf[case].data

# f2 = ax.fill_between(freq, up_bnd, 
#                       lo_bnd,
#                       alpha=.1, linewidth=0, zorder = 2, color = 'blue')

# case = 'VOLC'

# p3 = lfv_means_unf[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'red',
#         label = f'{case} unforced')

# up_bnd = lfv_means_unf[case].data + lfv_std_unf[case].data
# lo_bnd = lfv_means_unf[case].data - lfv_std_unf[case].data

# f3 = ax.fill_between(freq, up_bnd, 
#                       lo_bnd,
#                       alpha=.1, linewidth=0, zorder = 2, color = 'red')

# # plot confidence intervals
# [plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]

# plt.xlabel('Period (yr)')
# plt.ylabel("LFV")
# plt.title(f'MTM-SVD CESM LME {case}')

# ax.legend()

# save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
# name = 'lfv_ALL_unforced_VOLC_unforced_comp'

# save_fig = input("Save fig? (y/n):").lower()

# if save_fig == 'y':
#     plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
#     plt.savefig(save_path+name+'.svg', format = 'svg')

# else:
#     plt.show()


#%% Plot single member lines instead of shading 
# pick which periods to showcase (years)

whole_spec = False
xticks = [100,80,70,60,50,40,30,20]#,20,10,5,3,2]
if whole_spec:
    xticks = [100,80,70,60,50,40,30,20,10,5,3]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

plt.xscale('log')

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='both',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.3,1.0)
ax.tick_params(axis='x',which='both',direction='out',
               pad=15, labelrotation=45)
plt.minorticks_off()

case = 'ALL'
forc = 'forced'


if forc == 'forced':
    ds = lfv_by_case[case]
elif forc == 'unforced':
    ds = lfv_by_case_unf[case]
    
n_lines=ds.run.size
colormap = plt.cm.nipy_spectral #I suggest to use nipy_spectral, Set1,Paired
colors = colormap(np.linspace(0, 1, n_lines))
ax.set_prop_cycle('color', colors)

for run, run_ds in ds.groupby('run'):
    run_ds.plot(ax=ax,label=f'Ensemble member {run+1}',
                linestyle = '-.',
                linewidth = 1.0)

ds.mean(dim='run').plot(ax=ax,linewidth=3,color='black', label = 'mean')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]
plt.axvline(x = 1/34, color = 'red')
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME {forc} {case} single lines')

ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/NA/')

name = f'lfv_indiv_lines_{forc}_{case}'
if whole_spec:
    name = name + '_whole_spec'
    
save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    
