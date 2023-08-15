# Script for MultiTaper Method-Singular Value Decomposition (MTM-SVD) in python
#
# ------------------------------------------------------------------
#
# This script is a direct adaptation of the Matlab toolbox developed by
# Marco Correa-Ramirez and Samuel Hormazabal at 
# Pontificia Universidad Catolica de Valparaiso
# Escuela de Ciencias del Mar, Valparaiso, Chile
# and is available through 
# http://www.meteo.psu.edu/holocene/public_html/Mann/tools/tools.php
#
# This script was adapted by Mathilde Jutras at McGill University, Canada
# Copyright (C) 2020, Mathilde Jutras
# and is available under the GNU General Public License v3.0
# 
# The script may be used, copied, or redistributed as long as it is cited as follow:
# Mathilde Jutras. (2020, July 6). mathildejutras/mtm-svd-python: v1.0.0-alpha (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3932319
#
# This software may be used, copied, or redistributed as long as it is not 
# sold and that this copyright notice is reproduced on each copy made. 
# This routine is provided as is without any express or implied warranties.
#
# Questions or comments to:
# M. Jutras, mathilde.jutras@mail.mcgill.ca
#
# Last update:
# July 2020
#
# ------------------------------------------------------------------
#
# The script is structured as follows:
#
# In the main script is found in mtm-svd-python.py
# In the first section, the user can load the data,
# assuming the outputs are stored in a netcdf format.
# In the secton section, functions are called to calculate the spectrum
# The user will then be asked for which frequencies he wants to plot 
# the spatial patterns associated with the variability.
# In the third section, the spatial patterns are plotted and saved
#
# The required functions are found in mtm_functions.py
#
# ------------------------------------------------------------------
#

from mtm_functions_AF import *
import xarray as xr
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle as pkl

# start timer
start=datetime.now()
#

# -----------------
# 1) Load the data
# -----------------

#path to the climate dataset to be utilized
path = "//Volumes//AlejoED//Work//MannSteinman_Proj//Data//HadCRUT4_historic_data//HadCRUT4_temp_nc//"

files = listdir(path)   
files.sort()

print('Load data...')

dsl = xr.open_dataset(path+files[0])
dt = 1 # yearly data (monthly values averaged below)
lon = dsl.longitude
lat = dsl.latitude

for i in range(0, len(files)):
    print(i)
    files_nam = path + files[i]

ds = xr.open_mfdataset(files_nam)

files_num = len(files)

# Plot map of the variable
#xgrid, ygrid = np.meshgrid(lon,lat)
#plt.pcolor(xgrid, ygrid, var[0,:,:], cmap='jet')
#cbar=plt.colorbar()
#plt.show()

# -------------------
# 2) Compute the LVF
# -------------------

print('Apply the MTM-SVD...')

# Slepian tapers
nw = 2; # bandwidth
kk = 3; # number of orthogonal windows

# Reshape the 2d array to a 1d array
tas = ds.temperature_anomaly.values
tas_ts = tas.reshape((tas.shape[0],tas.shape[1]*tas.shape[2]), order='F')

#p, n = tas_ts.shape
years = ds.year.values

# calculate annual averages from monthly data
tas_ts_annual = annual_means(tas,years)

#number of years
n,p = tas_ts_annual.shape

#create meshgrid from latitude and longitude values
[x,y] = np.meshgrid(lon.longitude.values,lat.latitude.values)
#calculate weights matrix based on latitude
w = np.sqrt(np.cos(np.radians(y)));
w=w.reshape(1,w.shape[0]*w.shape[1],order='F')

# Compute the LFV
#[freq, lfv] = mtm_svd_lfv(tas_ts_annual,nw,kk,dt)
[freq, lfv] = mtm_svd_lfv(tas_ts_annual,nw,kk,dt,w)

# Compute the confidence intervals
niter = 1000    # minimum of 1000 iterations
sl = [.99,.95,.9,.8,.5]
[conffreq, conflevel, LFVs] = mtm_svd_conf(tas_ts_annual,nw,kk,dt,niter,sl,w)
conflevel = np.asarray(conflevel)

# calculate C.I. mean values for secular and non-secular bands
fr_sec = nw/(n*dt)
fr_sec_ind = np.where(conffreq < fr_sec)[0][-1]
ci_sec = np.nanmean(conflevel[:,0:fr_sec_ind],axis=1)
ci_nsec = np.nanmean(conflevel[:,fr_sec_ind+1:],axis=1)

# Adjust C.I values so they match
lfv_mean = np.nanmean(lfv[fr_sec_ind:])
mean_ci = ci_nsec[-1]
adj_factor = lfv_mean/mean_ci
adj_ci = np.array([ci_sec*adj_factor,ci_nsec*adj_factor])

# Plot the spectrum ___________________________________________
x = np.array([conffreq[0],conffreq[fr_sec_ind],conffreq[-1]])
fig, ax = plt.subplots()
ax.plot(freq,lfv)
plt.xlim([1/100., 1/2.])
plt.xlabel('Frequency [1/year]') ; plt.ylabel('LFV')

for i in range(0,len(sl)):
    y = np.array([adj_ci[0,i],adj_ci[1,i],adj_ci[1,i]])
    ax.plot(x,y)
    
fig.show
    
print(datetime.now()-start)

# save spectrum data to results folder
with open(f'results\\analyses\\hadcrut4_lfv_ci_freq_1000_{datetime.now().strftime("%b%d,%Y_%I:%M%p")}','wb') as f:
    pkl.dump([freq,lfv,conffreq,conflevel],f)


# fo = [float(each) for each in input('Enter the frequencies for which there is a significant peak and for which you want to plot the map of variance (separated by commas, no space):').split(',')]

# # --------------------------------
# # 3) Reconstruct spatial patterns
# # --------------------------------

# # Select frequency(ies) (instead of user-interaction selection)
# #fo = [0.02, 0.05, 0.15, 0.19, 0.24, 0.276, 0.38] 

# # Calculate the reconstruction

# vexp, totvarexp, iis = mtm_svd_recon(tas_ts_annual,nw,kk,dt,fo)

# # Plot the map for each frequency peak

# for i in range(len(fo)):

#  	RV = np.reshape(vexp[i],x.shape, order='F')

#  	fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[1,3]},figsize=(5,7))

#  	ax1.plot(freq, lfv)
#  	ax1.plot(conffreq, LFVs[0,:], '--', c='grey')
#  	ax1.plot(freq[iis[i]],lfv[iis[i]],'r*',markersize=10)
#  	ax1.set_xlabel('Frequency [1/years]')
#  	ax1.set_title('LVF at %i m')

#  	pc = ax2.pcolor(x, y, RV, cmap='jet', vmin=0, vmax=50) 
#  	cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
#  	cbar.set_label('Variance')
#  	ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

#  	plt.tight_layout()
#  	#plt.savefig('Figs/peak_analysis_%s_%im_%.2fyrs.jpg'%(model,d,1./fo[i]))
#  	#plt.show()
#  	plt.clf()
