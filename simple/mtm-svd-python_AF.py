
from mtm_funcs import *
import xarray as xr
from os import listdir
import os 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl

# %%-----------------
# 1) Load the raw data and read .nc file 
# -------------------

#path to the climate dataset to be utilized
path = "//Volumes//AlejoED//Work//MannSteinman_Proj//Data//HadCRUT4_historic_data//HadCRUT4_temp_nc//"
files = listdir(path)   
files.sort()

print('Load in data from NetCDF files...')

dsl = xr.open_dataset(path+files[0])
dt = 1 # yearly data (monthly values averaged below)
lon = dsl.longitude
lat = dsl.latitude

# Assign temperature data to 'tas' variable for analysis
tas = dsl.temperature_anomaly.values

# Assign time variables
time = dsl.time.values
years = dsl.year.values

# shape variables of data
time_steps, spatial_x, spatial_y = tas.shape

# %%-----------------
# 2) Compute the LFV spectrum of the dataset
# -------------------

print('Apply the MTM-SVD...')

# =============================================================================
# Values for MTM-SVD analysis
# =============================================================================
nw = 2; # bandwidth
kk = 3; # number of orthogonal windows

# =============================================================================
# Calculate annual means of monthly data
# =============================================================================

# Get rid of first and last years of data 
years_keep = np.unique(years)[1:-1] # array of years which will be kept (get rid of first and last)
ix = np.where((np.array(years) >= years_keep[0]) & (np.array(years) <= years_keep[-1])) # indexes where desired data is located in tas array
tas = tas[ix]

# Calculate the annual means
tas_reshaped = tas.reshape(-1,12,spatial_x,spatial_y)
tas_annual = np.mean(tas_reshaped, axis=1)

# Reshape the 3d array to a 2d array
tas = reshape_3d_to_2d(tas)

# Weights based on latitude
[xx,yy] = np.meshgrid(lon.longitude.values,lat.latitude.values)
w = np.sqrt(np.cos(np.radians(yy)));
w=w.reshape(1,w.shape[0]*w.shape[1],order='F') #reshape w array

# Compute the LFV
[freq, lfv] = mtm_svd_lfv(tas,nw,kk,dt,w)

# %%-----------------
# 4) Compute the confidece intervals for the reference data 
# -------------------

# =============================================================================
# Values for Confidence Interval calculation
# =============================================================================
niter = 10    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

# conflevels -> 1st column secular, 2nd column non secular (only nonsecular matters)
[conffreq, conflevels] = mtm_svd_conf(tas,nw,kk,dt,niter,sl,w) 

# =============================================================================
# Rescale Confidence Intervals to mean of reference LFV so 50% confidence interval
# matches mean value of the spectrum and all other values are scaled accordingly
# =============================================================================

# Rescaling of confidence intervals 
fr_sec = nw/(tas.shape[0]*dt) # secular frequency value
fr_sec_ix = np.where(freq < fr_sec)[0][-1] 

lfv_mean = np.nanmean(lfv[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 
mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
adj_ci = conflevels * adj_factor # adjustment for confidence interval values

# %%
# Plot the spectrum ___________________________________________
x_ci = np.array([conffreq[0],conffreq[fr_sec_ind],conffreq[-1]])
fig, ax = plt.subplots()
ax.plot(freq,lfv)
plt.xlim([1/100., 1/2.])
plt.xlabel('Frequency [1/year]') ; plt.ylabel('LFV')

for i in range(0,len(sl)):
    y_ci = np.array([adj_ci[0,i],adj_ci[1,i],adj_ci[1,i]])
    ax.plot(x_ci,y_ci)
    
fig.show

fig_name = f'results//LFV_plot//hadcrut4_lfv_ci_{niter}i_{datetime.now().strftime("%b%d,%Y_%I.%M%p")}.pdf'
os.makedirs(os.path.dirname(fig_name), exist_ok = True)
plt.savefig(fig_name)

    
print(datetime.now()-start)

# save spectrum data to results folder
file_name = f'results//LFV//hadcrut4_lfv_ci_1000_{datetime.now().strftime("%b%d,%Y_%I.%M%p")}'
os.makedirs(os.path.dirname(file_name), exist_ok = True)
with open(file_name,'wb') as f:
    pkl.dump([freq,lfv,conffreq,conflevel,x_ci,y_ci],f)
