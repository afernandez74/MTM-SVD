
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

year_i = 1150
year_f = 1350

# Select frequency(ies)

fo = 1/25
# location of gridcell to reconstruct
# rec_lon = 317.5 #labrador
# rec_lat = 54
rec_lon = 320
rec_lat = 55.89

case = 'ALL_FORCING'
unforced = True
run = 6
NA = True

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
        lon=(["lat", "lon"], xx),
        lat=(["lat", "lon"], yy)),
    attrs=dict(
        description=f"Variance explained by {1./fo:.2f}",
        units="%"), 
    name = f'Var exp {case}_iter{run:03} {1./fo:.2f} yr period'
)

Rec = np.reshape(R,tas_np.shape, order='F')

Rec = xr.DataArray(
    data = Rec,
    dims = tas.TS.dims,
    coords=tas.TS.coords,
    attrs=dict(
        description=f"Signal recons for freq={1./fo:.2f}",
        units="%"), 
    name = f'Sig recons {case}_iter{run:03} {1./fo:.2f} yr period'
)

Rec_loc = Rec.sel(lat = rec_lat, lon = rec_lon, method = 'nearest')

# =============================================================================
# spectrum
# =============================================================================
fig = plt.figure(figsize=(15,30))

ax1 = plt.subplot(311)
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
ax1.set_title(f'LFV spectrum {case}run{run}')
if unforced:
    ax1.set_title(f'LFV spectrum {case}run{run} unforced')

ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map 
# =============================================================================

ax2 = plt.subplot(312,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')

p = RV.plot.contourf(ax = ax2,
            add_colorbar = False,
            transform = ccrs.PlateCarree(),
            # vmin = 0, vmax = 2,
            robust = True,
            levels = 40,
            cmap = 'turbo')
ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)

# add separate colorbar
cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = '% Variance Explained')

p.axes.coastlines(color='lightgrey')
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

# plot location of point of signal reconstruction
plt.plot(rec_lon,rec_lat,
         marker='X', markersize = 20, 
         color = 'black', fillstyle = 'none',markeredgewidth = 3.0,
         transform = ccrs.Geodetic())

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)

# =============================================================================
# Signal reconstruction at highest var gridcell
# =============================================================================

ax1 = plt.subplot(313)
mn = tas.TS.sel(lon=rec_lon,lat = rec_lat, method = 'nearest').mean(dim = 'year').values
p1 = (Rec_loc+mn).plot(linewidth = 3, zorder = 2)
p2 = tas.sel(lon=rec_lon,lat = rec_lat, method = 'nearest').TS.plot(linewidth = 1, zorder = 1)

# =============================================================================
#     save
# =============================================================================
# save_fig = input("Save figs? (y/n):").lower()

# save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
# name = f'lfv_{case}_run_{run}_per_{1./fo:.0f}yr'
# if unforced:
#     name = name + 'unforced'


# if save_fig == 'y':
#     plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
#     plt.savefig(save_path+name+'.svg', format = 'svg')

# else:
#     plt.show()

#%% recons whole ensemble 

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 851
year_f = 1849

# Select frequency(ies)

fo = 0.0151367 #peak for run=4 of VOLC (~66yr)
# fo = 0.008 # peak for run=4 of VOLC (ENSO band)
# fo = 0.0170 #peak for ALL_FORCING ensemble (~58yr)

# location of gridcell to reconstruct
#VOLC LOCS
rec_lon = 352.5 #tropical north atlantic
rec_lat = 19.89 #north atlantic 

# rec_lon = 335
# rec_lat = 48.32

#ALL FORCING locs
# rec_lon = 302.5 #st pierre
# rec_lat = 44.53
# rec_lon = 317.5 #labrador
# rec_lat = 54

case = 'VOLC'
unforced = False

n_runs = len(CESM_LME[case].run)

Rec_loc_ensemb = []

# save_fig = input("Save figs? (y/n):").lower()

for run in range(0,n_runs):


# Calculate the reconstruction
    if unforced == True:
        dat = CESM_LME_unforced[case].sel(run=run)
    else:
        dat = CESM_LME[case].sel(run=run)
    
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
            lon=(["lat", "lon"], xx),
            lat=(["lat", "lon"], yy)),
        attrs=dict(
            description=f"Variance explained by {1./fo:.2f}",
            units="%"), 
        name = f'Var exp {case}_iter{run:03} {1./fo:.2f} yr period'
    )
    
    Rec = np.reshape(R,tas_np.shape, order='F')
    
    Rec = xr.DataArray(
        data = Rec,
        dims = tas.TS.dims,
        coords=tas.TS.coords,
        attrs=dict(
            description=f"Signal recons for freq={1./fo:.2f}",
            units="%"), 
        name = f'Sig recons {case}_iter{run:03} {1./fo:.2f} yr period'
    )
    
    Rec_loc = Rec.sel(lat = rec_lat, lon = rec_lon, method = 'nearest')
    Rec_loc_ensemb.append(Rec_loc)

    # =============================================================================
    # spectrum
    # =============================================================================
    fig = plt.figure(figsize=(15,30))
    
    ax1 = plt.subplot(311)
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
    ax1.set_title(f'LFV spectrum {case}run{run}')
    if unforced:
        ax1.set_title(f'LFV spectrum {case}run{run} unforced')
    
    ax1.set_xlabel('LFV')
    ax1.set_ylabel('Period (yr)')
    
    # =============================================================================
    # map 
    # =============================================================================
    
    ax2 = plt.subplot(312,projection = ccrs.Robinson(central_longitude = -90), facecolor= 'grey')
    
    p = RV.plot.contourf(ax = ax2,
                add_colorbar = False,
                transform = ccrs.PlateCarree(),
                # vmin = 0, vmax = 2,
                robust = True,
                levels = 40,
                cmap = 'turbo')
    ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)
    
    # add separate colorbar
    cb = plt.colorbar(p, orientation='horizontal', 
                      # ticks=[0,0.5,1,1.5,2],
                      pad=0.05,shrink=0.8,label = '% Variance Explained')
    
    p.axes.coastlines(color='lightgrey')
    gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')
    
    # plot location of point of signal reconstruction
    plt.plot(rec_lon,rec_lat,
             marker='X', markersize = 20, 
             color = 'black', fillstyle = 'none',markeredgewidth = 3.0,
             transform = ccrs.Geodetic())
    
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)
    
    # =============================================================================
    # Signal reconstruction at highest var gridcell
    # =============================================================================
    
    ax1 = plt.subplot(313)
    mn = tas.TS.sel(lon=rec_lon,lat = rec_lat, method = 'nearest').mean(dim = 'year').values
    p1 = (Rec_loc+mn).plot(linewidth = 3, zorder = 2)
    p2 = tas.sel(lon=rec_lon,lat = rec_lat, method = 'nearest').TS.plot(linewidth = 1, zorder = 1)
    
    # =============================================================================
    #     save
    # =============================================================================
    # save_fig = input("Save figs? (y/n):").lower()
    
    # save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
    # name = f'lfv_{case}_run_{run}_per_{1./fo:.0f}yr'
    # if unforced:
    #     name = name + 'unforced'
    
    
    # if save_fig == 'y':
    #     plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    #     plt.savefig(save_path+name+'.svg', format = 'svg')
    
    # else:
    #     plt.show()
    
#%% plot reconstruction ensemble

Rec_loc_ensemb = xr.concat(Rec_loc_ensemb, dim = 'run')

fig, ax = plt.subplots(figsize=(15, 10),
                       layout='constrained')

for run in range(0,n_runs):
    Rec_loc_ensemb.sel(run = run).plot(linewidth = 1, label = run)
Rec_loc_ensemb.mean(dim = 'run').plot(linewidth = 2, color = 'black', zorder = 10, label = 'mean')

ax.set_title(f'Reconstructed signals for MTM-SVD {case} analysis per = {1./fo:.2f} yrs\nIn location {rec_lon}° E, {rec_lat}° N')
if unforced:
    ax.set_title(f'Reconstructed signals for MTM-SVD {case} unforced analysis per = {1./fo:.2f} yrs\nIn location {rec_lon}° E, {rec_lat}° N')

ax.legend()
#%%
corrs = np.zeros((n_runs,n_runs))
for run_i in range (0,n_runs):
    rec_i = Rec_loc_ensemb.sel(run = run_i)
    for run_j in range (0,n_runs):
        rec_j = Rec_loc_ensemb.sel(run = run_j)
        corr_i = xr.corr(rec_i, rec_j)
        corrs[run_i,run_j]=corr_i.values
        
fig, ax = plt.subplots()
im = ax.imshow(corrs)
