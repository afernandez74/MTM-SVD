
from mtm_funcs import *
import xarray as xr
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


# %% 1) Load reference data and prepare datasets for analysis
 

#path to the climate dataset to be utilized
path = "had4_krig_v2_0_0.nc"
# files = listdir(path)   
# files.sort()

print('Load in data from NetCDF files...')

ds_monthly = xr.open_dataset(path)['temperature_anomaly']
dt = 1 # yearly data (monthly values averaged below)

# calculate annual means
ds = ds_monthly.groupby('time.year').mean(dim = 'time')

# obtain lat and lon
lon = ds.longitude
lat = ds.latitude

# Plot map of the variable
model = 'had4_obs_temp'
xgrid, ygrid = np.meshgrid(lon,lat)

ds.mean(dim='year').plot()
ds.mean(dim = ['latitude','longitude']).plot()

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

niter = 100    # Recommended -> 1000
sl = [.99,.95,.9,.5] # confidence levels

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
[conffreq, conflevels] = mtm_svd_conf(tas2d,nw,kk,dt,niter,sl,w) 

# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly

mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values


# %% Plot the spectrum ___________________________________________

# pick which periods to showcase (years)
xticks = [100,60,40,20,10,7,3,2]

# modify global setting
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 30

# figure drawing
fig = plt.figure(figsize=(20,12))
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
ax.yaxis.set_major_locator(MultipleLocator(0.1))

# plot line
p1 = ax.plot(freq,lfv,
        linestyle = '-',
        linewidth=2,
        zorder = 10,
        color = 'darkblue',
        label = 'LFV obs temp')

[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in adj_ci[:,1]]

#%% 5) Reconstruct spatial patterns

# Select frequency(ies)
fo = 0.016

# Calculate the reconstruction
R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas2d,nw,kk,dt,fo,w)

print(f'total variance explained by {fo} ={totvarexp}')

# Plot the map for each frequency peak

RV = np.reshape(vexp,xgrid.shape, order='F')
fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[1,3]},figsize=(10,16))
ax1.semilogx(freq, lfv, '-', c='k')
[plt.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in adj_ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=10)
ax1.set_xlabel('Frequency [1/years]')
# ax1.set_title('LVF at %i m'%d)

pc = ax2.pcolor(xgrid, ygrid, RV, cmap='jet', vmin=0) 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Variance')
# ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

plt.tight_layout()
# plt.savefig(f'Figs/{model}_peak_analysis_%s_%im_%.2fyrs.jpg'%(model,d,1./fo[i]))
plt.show()
# plt.clf()


print('finish')

