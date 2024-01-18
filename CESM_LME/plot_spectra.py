
# %% import functions and packages

from mtm_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#%% load data

path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')

lfv_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('lfv.nc')]
lfv_unforced_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('lfv_unforced')]
ci_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int')]

lfv = xr.open_dataset(path+lfv_file[0])
lfv_unforced = xr.open_dataset(path+lfv_unforced_file[0])
ci = np.load(path+ci_file[0])

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

# create lfv dictionary by case 

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

#%% plot spectra line only

freq = lfv['ALL_FORCING_001_lfv'].freq.data

# pick which periods to showcase (years)

xticks = [100,60,40,20]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,0.7)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

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


case = 'SOLAR'

p4 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'orange',
        label = 'SOLAR ONLY')


case = 'ORBITAL'

p5 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 5,
        color = 'purple',
        label = 'ORBITAL ONLY')


# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title('MTM-SVD CESM LME')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
name = 'lfv_all_spectra'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
#%% plot spectra with shaded spread

freq = lfv['CNTL_lfv'].freq.data

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

# pick which periods to showcase (years)

xticks = [100,60,40,20]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,0.8)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

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
        color = 'darkred',
        label = 'ALL FORCING')

up_bnd = lfv_means[case].data + lfv_std[case].data
lo_bnd = lfv_means[case].data - lfv_std[case].data

f2 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'darkred')

case = 'VOLC'

p3 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'VOLCANIC ONLY')

up_bnd = lfv_means[case].data + lfv_std[case].data
lo_bnd = lfv_means[case].data - lfv_std[case].data

f3 = ax.fill_between(freq, up_bnd, 
                      lo_bnd,
                      alpha=.1, linewidth=0, zorder = 2, color = 'darkblue')

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
plt.title('MTM-SVD CESM LME')
ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
name = f'lfv_ALL_plus_VOLC_shaded'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    

#%% Compare forced and unforced lines only

# pick which periods to showcase (years)

xticks = [100,60,40,20,10,5]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.4,0.8)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# plot lines

case = 'SOLAR'

p2 = lfv_means[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = f'{case} forced')

p3 = lfv_means_unf[case].plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'red',
        label = f'{case} unforced')

# case = 'VOLC'

# p3 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'red',
#         label = 'VOLCANIC ONLY')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]

plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME {case}')

ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
name = f'lfv_forced_unforced_{case}'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    
#%% Plot all lines instead of shading
# pick which periods to showcase (years)

# xticks = [100,60,40,20]
xticks = [10,7,6,5,4,3]

# figure drawing

fig = plt.figure(figsize=(30,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])

# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
# plt.ylim(0.4,0.7)
plt.ylim(0.4,1)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

# plot lines

case = 'VOLC'

ds = lfv_by_case[case]

for run, run_ds in ds.groupby('run'):
    run_ds.plot(label=run)
ds.mean(dim='run').plot(linewidth=2,color='black', label = 'mean')
# case = 'VOLC'

# p3 = lfv_means[case].plot(
#         linestyle = '-',
#         linewidth=2,
#         zorder = 10,
#         color = 'red',
#         label = 'VOLCANIC ONLY')

# plot confidence intervals
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5) for i in ci[:,1]]

plt.xlabel('Period (yr)')
plt.ylabel("LFV")
plt.title(f'MTM-SVD CESM LME {case} single lines')

ax.legend()

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
name = f'lfv_lines_{case}'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    
