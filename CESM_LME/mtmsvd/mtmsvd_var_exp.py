#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:21:53 2024

@author: afer
"""
# plot variance explained map for single frequency value
# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean

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

#choose confidence interval file

path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')

ci_file = [entry for entry in os.listdir(path) if not entry.startswith('.') and entry.startswith('conf_int')]
ci_file.sort()
print ('\n' + 'C.I files:')
for ix, string in enumerate(ci_file):
    print(f'{ix:2} : {string}')

index_ci = int(input('ci file index: '))#choose the index for confidence interval file 
ci_file_name = ci_file[index_ci][:ci_file[index_ci].rfind('.nc')]
ci = np.load(path+ci_file[index_ci])

# load North Atlantic mask file 

path = os.path.expanduser('~/mtm_local/CESM_LME/masks/NA_mask.nc')
NA_mask = xr.open_dataarray(path)

del ci_file

#%% Subtract ensemble means from each simulation 

CESM_LME_unforced = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        ens_mean = case_ds.mean(dim = 'run').expand_dims(run = case_ds['run'])

        CESM_LME_unforced[case] = case_ds - ens_mean

#%% Ensemble mean dictionary
CESM_LME_EM = {}

for case, case_ds in CESM_LME.items():

    if case != 'CNTL':

        CESM_LME_EM[case]  = case_ds.mean(dim = 'run')

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

#%% recons one simulation only

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 1300
year_f = 1800

# Select frequency(ies)

# fo = 0.0151367 #peak for run=4 of VOLC (~66yr)
# fo = 0.136963# peak for run=4 of VOLC (ENSO band)
# fo = 0.0170 #peak for ALL_FORCING ensemble (~58yr)
# fo = 0.023 #~43 yr
fo = 1/27

case = 'CNTL'
unforced =False
run = 6
NA = True

save_fig = input("Save figs? (y/n):").lower()

# Calculate the reconstruction
if case != 'CNTL':
    if unforced:
        dat = CESM_LME_unforced[case].sel(run=run)
    else:
        dat = CESM_LME[case].sel(run=run)
else:
    dat = CESM_LME[case]

dat = dat.sel(lat = slice(-60,60))
dat = dat.where(NA_mask == 1) if NA else dat


# Weights based on latitude
[xx,yy] = np.meshgrid(dat.lon,dat.lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# temp data
tas = dat.sel(year=slice(year_i,year_f))
tas_np = tas.TS.to_numpy()

# reshape data to 2d
tas_2d = reshape_3d_to_2d(tas_np)

freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
spectrum = xr.DataArray(lfv, coords = [freq], dims=['freq'])

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)

RV = np.reshape(vexp,xx.shape, order='F')

RV = xr.DataArray(
    data = RV,
    dims = ('lat','lon'),
    coords=dict(
        lon=tas.TS.lon.values,
        lat=tas.TS.lat.values),
    attrs=dict(
        description=f"Variance explained by {1./fo:.2f}",
        units="%"), 
    name = f'Var exp {case}_iter{run:03} {1./fo:.2f} yr period'
)


print(f'total variance explained by {fo} = {totvarexp}')

# PLOTS

# =============================================================================
# spectrum
# =============================================================================
fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot(211)
xticks = [100,80,70,60,50,40,30,20]

# figure drawing

ax1.set_xscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax1.set_xticklabels(xticks2_labels)
ax1.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
plt.ylim(0.3,1.0)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.minorticks_off()

p1 = spectrum.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue')

[ax1.axhline(y=i, color='black', linestyle='--', alpha=.8, zorder = 1) for i in ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=20, zorder = 20)
ax1.legend()
ax1.set_title(f'LFV spectrum {case}run{run}\n{year_i}-{year_f}')
if unforced:
    ax1.set_title(f'LFV spectrum {case}run{run} unforced\n{year_i}-{year_f}')

ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map 
# =============================================================================

ax2 = plt.subplot(212,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')

p = RV.plot.contourf(ax = ax2,
            add_colorbar = False,
            transform = ccrs.PlateCarree(),
            # vmin = 0, 
            # vmax = 18,
            robust = True,
            levels = 12,
            cmap = 'hot_r')
ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)

# add separate colorbar
cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

p.axes.coastlines(color='lightgrey')

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)

# =============================================================================
#     save
# =============================================================================
path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
path = (path + 'NA/') if NA else (path+'GLOB/')

name = f'lfv_{case}_run_{run}_per_{1./fo:.0f}yr'

name = name + '_unforced' if unforced else name
name = name + '_NA' if NA else name


if save_fig == 'y':
    plt.savefig(path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(path+name+'.svg', format = 'svg')

else:
    plt.show()
#%% Find highest mean var explained point 

# =============================================================================
#  find hightest mean var explained North Atlantic (above 50 deg)
# =============================================================================

# subselect lat slice 
RV = RV.sel(lat = slice(0,60), lon = slice(260,300))

RV_max = RV.where(RV==RV.max(), drop=True).squeeze()

# =============================================================================
# plot map with location 
# =============================================================================

fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                  facecolor='grey')

p = RV.plot.contourf(cmap = 'turbo',
                  add_colorbar = False,
                  transform = ccrs.PlateCarree(),
                  levels = 10,
                  robust = True
                  # vmin = 0, vmax = 2.0)
                  )

ax = p.axes.coastlines()
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')

plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs on {case} run {run}\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N',
          pad = 20)
if unforced:
    plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs on {case} run {run} unforced\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N\n{year_i}-{year_f}',
          pad = 20)


# plot location of highest variance explained
plt.plot(RV_max.lon.values, RV_max.lat.values,
          marker='X', markersize = 20, 
          color = 'black', fillstyle = 'none',markeredgewidth = 3.0,
          transform = ccrs.Geodetic())

#%% recons whole ensemble
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 1600
year_f = 1850

# Select frequency(ies)

# fo = 0.0151367 #peak for run=4 of VOLC (~66yr)
# fo = 0.15844 # peak for run=4 of VOLC (ENSO band)
# fo = 0.0170 #peak for ALL_FORCING ensemble (~58yr)
fo = 1/35

case = 'VOLC'
unforced = False
n_runs = len(CESM_LME[case].run)
NA = False

save_fig = input("Save figs? (y/n):").lower()

for run_i in range(0,n_runs):
    
    # Calculate the reconstruction
    if unforced:
        dat = CESM_LME_unforced[case].sel(run=run_i)
    else:
        dat = CESM_LME[case].sel(run=run_i)
        
    dat = dat.where(NA_mask == 1) if NA else dat

    
    # Weights based on latitude
    [xx,yy] = np.meshgrid(dat.lon,dat.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')
    
    # temp data
    tas = dat.sel(year=slice(year_i,year_f))
    tas_np = tas.TS.to_numpy()
    
    # reshape data to 2d
    tas_2d = reshape_3d_to_2d(tas_np)
    
    freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
    spectrum = xr.DataArray(lfv, coords = [freq], dims=['freq'])
    
    R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)
    
    RV = np.reshape(vexp,xx.shape, order='F')
    
    RV = xr.DataArray(
        data = RV,
        dims = ('lat','lon'),
        coords=dict(
            lon=tas.TS.lon.values,
            lat=tas.TS.lat.values),
        attrs=dict(
            description=f"Variance explained by {1./fo:.2f}",
            units="%"), 
        name = f'Var exp {case}_iter{run:03} {1./fo:.2f} yr period'
    )
    
    
    print(f'total variance explained by {fo} = {totvarexp}')
    
    # PLOTS
    
    # =============================================================================
    # spectrum
    # =============================================================================
    fig = plt.figure(figsize=(15,15))
    
    ax1 = plt.subplot(211)
    xticks = [100,80,60,40,30,10,5,3]

    
    # figure drawing
    
    ax1.set_xscale('log')
    # set x ticks and labels
    
    xticks2 = [1/x for x in xticks]
    ax1.set_xticks(xticks2)
    xticks2_labels = [str(x) for x in xticks]
    ax1.set_xticklabels(xticks2_labels)
    ax1.grid(True,which='major',axis='both')
    plt.xlim((xticks2[0],xticks2[-1]))
    # plt.ylim(0.4,0.8)
    ax1.tick_params(axis='x',which='major',direction='out',
                   pad=15, labelrotation=45)
    ax1.minorticks_off()
    
    p1 = spectrum.plot(
            linestyle = '-',
            linewidth=2,
            zorder = 10,
            color = 'blue')
    
    [ax1.axhline(y=i, color='black', linestyle='--', alpha=.8, zorder = 1) for i in ci[:,1]]
    ax1.plot(freq[iif],lfv[iif],'r*',markersize=20, zorder = 20)
    ax1.legend()
    ax1.set_title(f'LFV spectrum {case}run{run_i}\n{year_i}-{year_f}')
    if unforced:
        ax1.set_title(f'LFV spectrum {case}run{run_i} unforced\n{year_i}-{year_f}')
        
    ax1.set_xlabel('LFV')
    ax1.set_ylabel('Period (yr)')
    
    # =============================================================================
    # map 
    # =============================================================================
    
    ax2 = plt.subplot(212,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')
    
    p = RV.plot.contourf(ax = ax2,
                add_colorbar = False,
                transform = ccrs.PlateCarree(),
                # vmin = 0, vmax = 2.0,
                robust = True,
                levels = 40,
                cmap = 'turbo')
    ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)
    
    # add separate colorbar
    cb = plt.colorbar(p, orientation='horizontal', 
                      # ticks=[0.0,0.5,1.0,1.5,2],
                      pad=0.05,shrink=0.8,label = '% Variance Explained')
    
    p.axes.coastlines(color='lightgrey')
    gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)
    
    # =============================================================================
    #     save
    # =============================================================================
    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
    name = f'lfv_{case}_run_{run}_per_{1./fo:.0f}yr'
    if unforced:
        name = name + 'unforced'
    
    
    if save_fig == 'y':
        plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
        plt.savefig(save_path+name+'.svg', format = 'svg')
    
    else:
        plt.show()
        
        

#%% average var exp map

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 1150
year_f = 1350

# Select frequency(ies)

# fo = 0.0151367 #peak for run=4 of VOLC (~66yr)
# fo = 0.25 # peak for run=4 of VOLC (ENSO band)
fo =1/60

case = 'ALL_FORCING'
unforced = True
NA = False

spectra = {}
RV = []

for run_i in CESM_LME[case].run:

    # Calculate the reconstruction
    if unforced == True:
        dat = CESM_LME_unforced[case].sel(run=run_i)
    else:
        dat = CESM_LME[case].sel(run=run_i)

    # temp data
    dat = dat.where(NA_mask == 1) if NA else dat

    # Weights based on latitude
    [xx,yy] = np.meshgrid(dat.lon,dat.lat)
    w = np.sqrt(np.cos(np.radians(yy)));
    w = w.reshape(1,w.shape[0]*w.shape[1],order='F')


    tas = dat.sel(year=slice(year_i,year_f))
    tas_np = tas.TS.to_numpy()
    
    # reshape data to 2d
    tas_2d = reshape_3d_to_2d(tas_np)
    
    freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
    spectra [f'{case}_{run_i:03}']= xr.DataArray(lfv, coords = [freq], dims=['freq'])
    
    R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)
    
    RV_i = np.reshape(vexp,xx.shape, order='F')
    
    RV_i = xr.DataArray(
        data = RV_i,
        dims = ('lat','lon'),
        coords=dict(
            lon=tas.TS.lon.values,
            lat=tas.TS.lat.values),
        attrs=dict(
            description=f"Variance explained by {1./fo:.2f}",
            units="%"), 
        name = f'Var exp {case}_iter{run_i:03} {1./fo:.2f} yr period'
    )
    
    RV.append(RV_i)
    
RV = xr.concat(RV, dim = 'run')

RV_mean = RV.mean(dim = 'run')


fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='grey')

p = RV_mean.plot.contourf(cmap = 'hot_r',
                 add_colorbar = False,
                 transform = ccrs.PlateCarree(),
                 levels = 10,
                 # vmin = 0, 
                 # vmax = 14.0,
                    vmax = 8,
                  robust = True,
                 )

ax = p.axes.coastlines()

cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  # pad=0.05,shrink=0.8,
                  label = '% Variance Explained')

plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case}\n{year_i}-{year_f}',pad = 20)
if unforced:
    plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case} unforced\n{year_i}-{year_f}',pad = 20)

p.axes.coastlines()

gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

# =============================================================================
# save
# =============================================================================
save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/GLOB/')
if NA:
    save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/NA/')

name = f'lfv_{case}_MEAN_VAR_EXP_per_{1./fo:.0f}yr_{year_i}-{year_f}'

name = name + '_NA' if NA else name
name = name + '_unforc' if unforced else name


save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()


#%% find hightest mean var explained point in map
# separate data for North Atlantic only

RV_mean_NA = RV_mean.sel(lat = slice(0,60), lon = slice(290,360))

# find max Var Exp in NA
RV_max = RV_mean_NA.where(RV_mean_NA==RV_mean_NA.max(), drop=True).squeeze()

# =============================================================================
# plot map with location 
# =============================================================================

fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='grey')

p = RV_mean_NA.plot.contourf(cmap = 'turbo',
                 add_colorbar = False,
                 transform = ccrs.PlateCarree(),
                 levels = 40,
                 vmin = 0, vmax = 2.0)

ax = p.axes.coastlines()
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
cb = plt.colorbar(p, orientation='horizontal', 
                  ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')

plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case}\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N',
          pad = 20)

if unforced:
    plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case} unforced\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N',
          pad = 20)

# plot location of highest variance explained
plt.plot(RV_max.lon.values, RV_max.lat.values,
         marker='X', markersize = 20, 
         color = 'black', fillstyle = 'none',markeredgewidth = 3.0,
         transform = ccrs.Geodetic())

#%% find hightest mean var explained North Atlantic (above 50 deg)
# separate data for North Atlantic only

RV_mean_NA = RV_mean.sel(lat = slice(50,60), lon = slice(290,360))

# find max Var Exp in NA
RV_max = RV_mean_NA.where(RV_mean_NA==RV_mean_NA.max(), drop=True).squeeze()

# =============================================================================
# plot map with location 
# =============================================================================

fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='grey')

p = RV_mean_NA.plot.contourf(cmap = 'turbo',
                 add_colorbar = False,
                 transform = ccrs.PlateCarree(),
                 levels = 40,
                 vmin = 0, vmax = 2.0)

ax = p.axes.coastlines()
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
cb = plt.colorbar(p, orientation='horizontal', 
                  ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')

plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case}\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N',
          pad = 20)
if unforced:
    plt.title(f'Mean Variance explained by period {1./fo:.2f} yrs over all runs of {case} unforced\nLocation of highest var exp = {RV_max.lon.values:.2f}°E, {RV_max.lat.values:.2f}°N',
          pad = 20)

# plot location of highest variance explained
plt.plot(RV_max.lon.values, RV_max.lat.values,
         marker='X', markersize = 20, 
         color = 'black', fillstyle = 'none',markeredgewidth = 3.0,
         transform = ccrs.Geodetic())

#%% recons ensemble mean 

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 1350
year_f = 1600

# Select frequency(ies)

fo = 1/30
case = 'ALL_FORCING'
NA = True


# Calculate the reconstruction

dat = CESM_LME_EM[case]

dat = dat.where(NA_mask == 1) if NA else dat


# Weights based on latitude
[xx,yy] = np.meshgrid(dat.lon,dat.lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# temp data
tas = dat.sel(year=slice(year_i,year_f))
tas_np = tas.TS.to_numpy()

# reshape data to 2d
tas_2d = reshape_3d_to_2d(tas_np)

freq, lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
spectrum = xr.DataArray(lfv, coords = [freq], dims=['freq'])

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)

RV = np.reshape(vexp,xx.shape, order='F')

RV = xr.DataArray(
    data = RV,
    dims = ('lat','lon'),
    coords=dict(
        lon=tas.TS.lon.values,
        lat=tas.TS.lat.values),
    attrs=dict(
        description=f"Variance explained by {1./fo:.2f}",
        units="%"), 
    name = f'Var exp {case} {1./fo:.2f} yr period'
)


print(f'total variance explained by {fo} = {totvarexp}')

# PLOTS

# =============================================================================
# spectrum
# =============================================================================
fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot(211)
xticks = [100,80,60,40,30,20]

# figure drawing

ax1.set_xscale('log')
# set x ticks and labels

xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax1.set_xticklabels(xticks2_labels)
ax1.grid(True,which='major',axis='both')
plt.xlim((xticks2[0],xticks2[-1]))
# plt.ylim(0.4,0.8)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.minorticks_off()

p1 = spectrum.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue')

[ax1.axhline(y=i, color='black', linestyle='--', alpha=.8, zorder = 1) for i in ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=20, zorder = 20)
ax1.legend()
ax1.set_title(f'LFV ensemble mean spectrum {case}\n{year_i}-{year_f}')
ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map 
# =============================================================================

ax2 = plt.subplot(212,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')

p = RV.plot.contourf(ax = ax2,
            add_colorbar = False,
            transform = ccrs.PlateCarree(),
            # vmin = 0, 
            # vmax = 2,
            robust = True,
            levels = 40,
            cmap = 'turbo')
ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)

# add separate colorbar
cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

p.axes.coastlines(color='lightgrey')

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)

# =============================================================================
#     save
# =============================================================================
save_fig = input("Save figs? (y/n):").lower()

path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
path = (path + 'NA/') if NA else (path+'GLOB/')

name = f'lfv_EM_{case}_per_{1./fo:.0f}yr'

name = name + '_NA' if NA else name


if save_fig == 'y':
    plt.savefig(path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(path+name+'.svg', format = 'svg')

else:
    plt.show()
