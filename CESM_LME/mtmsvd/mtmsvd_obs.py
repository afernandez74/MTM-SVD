# %% import functions and packages

from mtmsvd_funcs import *
import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean
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


#%% Load datasets 
# path to the climate dataset to be utilized
path = "/Users/afer/mtm_local/misc/hadcrut5/HadCRUT5_anom_ens_mean.nc"
# files = listdir(path)   
# files.sort()

print('Load in data from NetCDF files...')

ds_monthly = xr.open_dataset(path)['tas_mean']
dt = 1 # yearly data (monthly values averaged below)

# calculate annual means
ds = ds_monthly.groupby('time.year').mean(dim = 'time')

# obtain lat and lon
lon = ds.longitude
lat = ds.latitude

# Plot map of the variable
model = 'had5_obs_temp'
xgrid, ygrid = np.meshgrid(lon,lat)

fig = plt.figure(figsize=[15,15])
ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='grey')

RV = ds.polyfit('year',deg = 0).polyfit_coefficients.sel(degree=0)

p = RV.plot.contourf(cmap = 'turbo',
                 add_colorbar = False,
                 transform = ccrs.PlateCarree(),
                 levels = 40,
                 # vmin = 0, vmax = 2.0
                 )

ax = p.axes.coastlines()

gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')

cb = plt.colorbar(p, orientation='horizontal', 
                  # ticks=[0,0.5,1,1.5,2],
                  pad=0.05,shrink=0.8,label = 'Temperature anomaly')

plt.title('Rate of warming observed')

save_path = os.path.expanduser('~/mtm_local/SRS/figs/')
name = 'obs_warming'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()


# timesries
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ds.sel(latitude = slice(-60,60)).mean(dim = ['latitude','longitude']).plot()
plt.title('global mean temperature observation')

name = 'global_temp_anom'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()


#%% MTM_SVD analysis 
# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================

nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

# =============================================================================
# loop through all cases
# =============================================================================
# Weights based on latitude
[xx,yy] = np.meshgrid(lon,lat)
w = np.sqrt(np.cos(np.radians(yy)));
w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# temp data
tas_np = ds.to_numpy()
tas_2d = reshape_3d_to_2d(tas_np)
freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)

spectrum = xr.DataArray(
    data = lfv,
    dims = 'freq',
    coords = [freq],
    name = f'HadCRUT5_lfv'
    )

LFV_obs = spectrum

#save results dataset as .nc file
path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv/")
LFV_obs.to_netcdf(path+'HadCRUT5_lfv.nc')

#%%plot spectrum
freq = LFV_obs.freq

# pick which periods to showcase (years)

# pick which periods to showcase (years)
whole_spec = True
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

p2 = LFV_obs.plot(
        linestyle = '-',
        linewidth=3,
        zorder = 10,
        color = 'darkgreen',
        label = 'ALL FORCING')

ci = np.load(os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/conf_int_HadCRUT5.npy'))
[plt.axhline(y=i, color='black', linestyle='--', linewidth = 1.5) for i in ci[:,1]]
plt.xlabel('Period (yr)')
plt.ylabel("LFV")

name = 'obs_LFV'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    



# %% Confidence intervals (CNTL)

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.5] # confidence levels

# conflevels -> 1st column secular, 2nd column non secular 
print(f'Confidence Interval Calculation for CNTL case ({niter} iterations)...')
[conffreq, conflevels] = mtm_svd_conf(tas_2d,nw,kk,dt,niter,sl,w)

fr_sec = nw/(tas_2d.shape[0]*dt) # secular frequency value
fr_sec_ix = np.where(conffreq < fr_sec)[0][-1] 

# renormalize confidence intervals so that mean nonsecular value matches mean of spectrum

lfv_mean = LFV_obs[fr_sec_ix:].mean().values
ci_mean = conflevels[-1,-1]

adj_fac = lfv_mean/ci_mean

conflevels[:,1]=conflevels[:,1]*adj_fac

# save results
np.save(path+'conf_int_HadCRUT5.npy',conflevels)

