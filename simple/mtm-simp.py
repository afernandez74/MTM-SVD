
from mtm_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid


# %% 1) Load reference data and prepare datasets for analysis
 

#path to the climate dataset to be utilized
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

RV = ds.mean(dim='year')
fig, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(15,16))
ax2.coastlines()
pc = ax2.pcolor(xgrid, ygrid, RV, cmap='jet') 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Temperature anomaly')
plt.title('map of observed warming')


# timesries
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ds.mean(dim = ['latitude','longitude']).plot()
plt.title('global mean temperature observation')

# %% 2) Compute the LFV spectrum of the reference data
# -------------------

print('Apply the MTM-SVD...')

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================
nw = 2; # bandwidth (p)
kk = 3; # number of orthogonal windows (S)

# =============================================================================
# Calculate annual means of monthly data
# =============================================================================

ds = ds.isel(year = slice(1,-1)) # Get rid of first and last years of data 

# get temperature data and reshape it 
tas = ds.to_numpy()
tas2d = reshape_3d_to_2d(tas)
N = tas2d.shape[0] # Number of years


# Weights based on latitude
w = np.sqrt(np.cos(np.radians(ygrid)));
w=w.reshape(1,w.shape[0]*w.shape[1],order='F') #reshape w array

# Compute the LFV
[freq, lfv] = mtm_svd_lfv(tas2d,nw,kk,dt,w)

fr_sec = nw/(N*dt) # secular frequency value (1/year)
fr_sec_ix = np.where(freq < fr_sec)[0][-1] # index of secular frequency in freq array

lfv_mean = np.nanmean(lfv[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 

spectrum = xr.DataArray(lfv, coords = [freq], dims=['freq'])

# %% 3) Compute the confidece intervals for the reference data 
# -------------------

# Values for Confidence Interval calculation

niter = 1000    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
[conffreq, conflevels] = mtm_svd_conf(tas2d,nw,kk,dt,niter,sl,w) 

# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly

mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values

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


# %% Plot the spectrum ___________________________________________

xticks = [np.floor(1/fr_sec),50,40,30,20,10,7,5,4,3,2]

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
# plt.ylim(0.4,1.0)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
plt.minorticks_off()
p1 = spectrum.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = 'HadCRUT5 obs temp anom')

[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in adj_ci[:,1]]
plt.title('LFV spectrum of temperature observations')
plt.xlabel('LFV')
plt.ylabel('Period (yr)')

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/lfv/')
name = f'lfv_hadcrut5_nw_{nw}'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()

#%% 5) Reconstruct spatial patterns

# Select frequency(ies)

# fo = 0.01953125 #~50 yr peak
fo = 0.2734375 #ENSO PEAK
# Calculate the reconstruction
R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas2d,nw,kk,dt,fo,w)

print(f'total variance explained by {fo} ={totvarexp}')

#field of variance explained variable
RV = np.reshape(vexp,xgrid.shape, order='F')

RV = xr.DataArray(
    data = RV,
    dims = ('lat','lon'),
    coords=dict(
        lon=(["lat", "lon"], xgrid),
        lat=(["lat", "lon"], ygrid)),
    attrs=dict(
        description=f"Variance explained by {1./fo:.2f}",
        units="%"), 
    name = f'Var exp {1./fo:.2f} yr period'
)


# %%Plot the map for each frequency peak

# =============================================================================
# spectrum
# =============================================================================

fig = plt.figure(figsize=(15,15))

ax1 = plt.subplot(211)
xticks = [np.floor(1/fr_sec),80,60,40,30,20,10,7,5,4,3,2]

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

p1 = spectrum.plot(
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'blue',
        label = 'HadCRUT5 obs temp anom')

[ax1.axhline(y=i, color='black', linestyle='--', alpha=.8, zorder = 1) for i in adj_ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=20, zorder = 20)
ax1.legend()
ax1.set_title('LFV spectrum of temperature observations')
ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map 
# ========================================================s====================
ax2 = plt.subplot(212,projection = ccrs.Robinson(), facecolor= 'grey')

p = RV.plot(ax = ax2,
            add_colorbar = False,
            transform = ccrs.PlateCarree(),
            vmin = 0, vmax = 10,
            cmap = 'turbo')
ax2.set_title(f'Variance explained by period {1./fo:.2f} yrs = {totvarexp:.2f}%',pad = 20,)

# add separate colorbar
cb = plt.colorbar(p, orientation='horizontal', pad=0.05,shrink=0.8,label = '% Variance Explained')

p.axes.coastlines()

plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=5.0)

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/var_exp/')
name = f'lfv_hadcrut5_per_{(1./fo):.0f}yr'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
