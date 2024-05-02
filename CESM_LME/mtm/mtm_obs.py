import pyleoclim as pyleo
import numpy as np
import xarray as xr
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import geopandas as gpd
from descartes import PolygonPatch
from shapely.geometry import Point, Polygon
from North_Atlantic_mask_funcs import load_shape_file, select_shape, serial_mask
pyleo.set_style(style = 'journal',font_scale = 2.0, dpi =300)


#%% Load datasets 
# path to the climate dataset to be utilized
path = "/Users/afer/mtm_local/misc/hadcrut5/HadCRUT5_anom_ens_mean.nc"
# files = listdir(path)   
# files.sort()

print('Load in data from NetCDF files...')

ds_monthly = xr.open_dataset(path)
dt = 1 # yearly data (monthly values averaged below)

# calculate annual means
ds = ds_monthly.groupby('time.year').mean(dim = 'time')

# obtain lat and lon
lon = ds.longitude
lat = ds.latitude

# Plot map of the variable
model = 'had5_obs_temp'
xgrid, ygrid = np.meshgrid(lon,lat)

RV = ds.mean(dim=['year','bnds']).tas_mean
fig, ax2 = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},figsize=(15,16))
ax2.coastlines()
pc = ax2.pcolor(xgrid, ygrid, RV, cmap='jet') 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Temperature anomaly')
plt.title('map of observed warming')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/map_warming')
plt.savefig(save_path, format = 'svg')


# timesries
fig = plt.figure(figsize=(20,12))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ds.mean(dim = ['latitude','longitude','bnds']).tas_mean.plot()
plt.title('global mean temperature observation')
save_path = os.path.expanduser('~/mtm_local/AGU23_figs/ts_warming')
plt.savefig(save_path, format = 'svg')
path = os.path.expanduser("~/mtm/MTM-SVD/simple/had4_krig_v2_0_0.nc")

#%% load North Atlantic mask 

shpfile = load_shape_file('/Users/afer/mtm_local/misc/World_Seas_IHO_v3/World_Seas_IHO_v3.shp')

NA = select_shape(shpfile, 'NAME', 'North Atlantic Ocean', plot=False)

ds_mask = ds.squeeze().sel(year = 1850,bnds = 1)

finalMask = xr.full_like(ds_mask, np.nan)

polygon = NA

# longitude/latitude for your data.
[xx,yy] = np.meshgrid(ds.longitude,ds.latitude)

print("masking North Atlantic")

temp_mask = serial_mask(xx, yy, polygon)

# set integer for the given region.
temp_mask = xr.DataArray(temp_mask, dims=['latitude', 'longitude']) # dims should be like your base data array you'll be masking

# Assign NaNs outside of mask, and index within
temp_mask = (temp_mask.where(temp_mask)).fillna(0)

# Add masked region to master array.
finalMask = finalMask.fillna(0) + temp_mask

# Make your zeros NaNs again.
finalMask = finalMask.where(finalMask > 0)

NA_mask = finalMask

del finalMask, NA, polygon,shpfile, temp_mask, xx, yy

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
# obs temp anomaly GLOBAL
# =============================================================================
dat = ds.tas_mean.mean(dim = ['latitude','longitude'])

series_GLOB = pyleo.Series(
    time=dat.year.values,
    value=dat.values,
    time_name="years AD",
    time_unit="yr",
    value_name="GMSAT anom",
    value_unit="K",
)
# =============================================================================
# obs temp anomaly NORTH ATLANTIC
# =============================================================================
dat_NA = ds.where(NA_mask ==1).tas_mean.mean(dim = ['latitude','longitude'])

series_NA = pyleo.Series(
    time=dat_NA.year.values,
    value=dat_NA.values,
    time_name="years AD",
    time_unit="yr",
    value_name="GMSAT anom",
    value_unit="K")



#%% some plots
nw = 2
N = len(dat.year)
npad = 2**int(np.ceil(np.log2(abs(N)))+2)

psd_GLOB_mtm = series_GLOB.detrend().spectral(method = 'mtm', settings = {
    'NW' : nw, 'nfft':npad})
psd_NA_mtm = series_NA.detrend().spectral(method = 'mtm', settings = {
    'NW' : nw, 'nfft':npad})

sig_n = 10000
qs = [0.50,0.90]

signif_GLOB_mtm = series_GLOB.standardize().detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'nfft':npad}).signif_test(
    number=sig_n, qs=qs)
signif_GLOB_series = signif_GLOB_mtm.signif_qs.psd_list

signif_NA_mtm = series_NA.standardize().detrend().spectral(method='mtm', settings = {
    'NW' : nw, 'nfft':npad}).signif_test(
    number=sig_n, qs=qs)
signif_NA_series = signif_NA_mtm.signif_qs.psd_list


#%% plot psd 

fig, ax = signif_GLOB_mtm.plot(figsize = (30,10),signif_linewidth=2, signif_clr='black',linewidth=2.0,
                          title = f'mtm analysis OBS TEMP GLOBAL nw={nw}')

fig, ax = signif_NA_mtm.plot(figsize = (30,10),signif_linewidth=2, signif_clr='black',linewidth=2.0,
                          title = f'mtm analysis OBS TEMP NORTH ATLANTIC nw={nw}')

# name = f'psd_mtm_nw{nw}_forced_VOLC_NA'

# save_fig = input("Save fig? (y/n):").lower()

# if save_fig == 'y':
#     plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
#     plt.savefig(save_path+name+'.svg', format = 'svg')

# else:
#     plt.show()

#%% plot AMO index 

series_NA.detrend(method = 'linear').filter(cutoff_freq = 1/10).plot(
    figsize = (30,10),title = 'AMV index for HADCRUT5 observational dataset')

#%% arrays spectrum significance
psd_GLOB_np = psd_GLOB_mtm.amplitude
psd_NA_np = psd_NA_mtm.amplitude

signif_GLOB_50 = signif_GLOB_series[0].amplitude
signif_GLOB_90 = signif_GLOB_series[1].amplitude

signif_NA_50 = signif_NA_series[0].amplitude
signif_NA_90 = signif_NA_series[1].amplitude
#%% ensembles mtm spectra, (minus 50% confidence)(divided by 90% confidence)

# pick which periods to showcase (years)
freq = psd_NA_mtm.frequency
xticks = [N/2,80,70,60,50,40,30,20,10,5,3,2]

# figure drawing 
fig = plt.figure(figsize=(25,10))

ax = fig.add_axes([0.1,0.1,0.5,0.8])
plt.xscale('log')

plt.minorticks_off()


# set x ticks and labels
xticks2 = [1/x for x in xticks]
ax.set_xticks(xticks2)
xticks2_labels = [str(x) for x in xticks]
ax.set_xticklabels(xticks2_labels)
ax.grid(True,which='major',axis='both')
ax.set_xlim((xticks2[0],xticks2[-1]))
# plt.xlim(1/30,1/2)
# plt.ylim(-1.0,1.5)
ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)

l1 = (psd_GLOB_np - signif_GLOB_50)/(signif_GLOB_90-signif_GLOB_50)
l2 = (psd_NA_np - signif_NA_50)/(signif_NA_90-signif_NA_50)

p1 = ax.plot(freq,l1, color = 'blue', label = 'Global', linewidth = 2.0, zorder = 5)

p2 = ax.plot(freq,l2, color = 'red', label = 'North Atlantic', linewidth = 2.0, zorder = 5)

[plt.axhline(y=i, color='black', linestyle='--', alpha=.8, linewidth = 1.5, zorder = 2) for i in [0,1]]
plt.legend()
plt.title(f'NA_MTM_forced_nw={nw}')
