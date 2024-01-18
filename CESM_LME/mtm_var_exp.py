#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:21:53 2024

@author: afer
"""

#%% recons 
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data 

year_i = 851
year_f = 1849

# Select frequency(ies)

# fo = 0.01513 #peak for run=4 of VOLC (~66yr)
fo = 0.15844 # peak for run=4 of VOLC (ENSO band)

# Calculate the reconstruction
dat = CESM_LME['VOLC'].sel(run=4)

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

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)

print(f'total variance explained by {fo} ={totvarexp}')

# Plot the map for each frequency peak

RV = np.reshape(vexp,xx.shape, order='F')
fig, (ax1, ax2) = plt.subplots(2,1,gridspec_kw={'height_ratios':[1,1]},subplot_kw={'projection': ccrs.PlateCarree()},figsize=(20,16))
ax1.plot(freq, lfv, '-', c='k')
# [ax1.axhline(y=i, color='black', linestyle='--', alpha=.8) for i in adj_ci[:,1]]
ax1.plot(freq[iif],lfv[iif],'r*',markersize=10)
ax1.set_xlabel('Frequency [1/years]')
# ax1.set_title('LVF at %i m'%d)

ax2.coastlines()
pc = ax2.pcolor(xx, yy, RV, cmap='jet', vmin=0) 
cbar = fig.colorbar(pc, ax=ax2, orientation='horizontal', pad=0.1)
cbar.set_label('Variance explained')
# ax2.set_title('Variance explained by period %.2f yrs'%(1./fo[i]))

# # plt.tight_layout()
# save_name = os.path.expanduser('~/mtm_local/AGU23_figs/CESM_recons_map_nino')
# plt.savefig(save_name, format = 'svg')
# plt.title('CESM model variance explained by period %.0f yrs'%(1./fo))

plt.show()
# plt.clf()