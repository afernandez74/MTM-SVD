#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:31:27 2024

@author: afer
"""

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=8),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=7),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=6),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=5),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=4),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=3),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=2),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=1),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(finalMask_OCEANS.sel(ocean=0),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(OCEANS_mask,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
OCEANS_mask
NA_mask = finalMask_NA
OCEANS_mask = finalMask_OCEANS.any(dim='ocean')
OCEANS_mask
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(OCEANS_mask,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
OCEANS_mask.values
temp=OCEANS_mask.values
OCEANS_mask = finalMask_OCEANS
OCEANS_mask
finalMask_OCEANS
temp = OCEANS_mask.values
ds
OCEANS_mask = OCEANS_mask.any(dim='ocean')
temp = OCEANS_mask.values
any_one = (finalMask_OCEANS == 1.0).any(dim='ocean')
OCEANS_mask = any_one.isel(ocean=slice(None))
finalMask_OCEANS = finalMask_OCEANS.where(finalMask_OCEANS > 0)
any_one = (finalMask_OCEANS == 1.0).any(dim='ocean')
OCEANS_mask = any_one.isel(ocean=slice(None))
temp = any_one
temp = any_one.values
OCEANS_mask = (finalMask_OCEANS == 1.0).any(dim='ocean')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(OCEANS_mask,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
freq = 1/10 #filter frequency for all data
corr_period = slice(1850,2020) #period of analysis for all

dat_glob = ds.where(
    OCEANS_mask==True).sel(
    latitude=slice(-60,60))
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(dat_glob.mean(dim = 'year'),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),d
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
runcell('Gloabal SST anomaly (60S - 60N)', '/Users/afer/.spyder-py3/autosave/reg_index.py')
dat_glob_ts.plot()
dat_glob_ts_filt = butter_lowpass(dat_glob_ts.values)
dat_glob_ts_filt = butter_lowpass(
    dat_glob_ts.values, cutoff_frequency=freq,sampling_frequency=1)
dat_glob_ts_filt = xr.DataArray(data = butter_lowpass(
    dat_glob_ts.values, cutoff_frequency=freq,sampling_frequency=1),
    dims = dat_glob_ts.dims,
    coords = dat_glob_ts.coords)
dat_glob_ts_filt.plot()
G = dat_glob_ts_filt
runcell('AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('Gloabal SST anomaly (60S - 60N)', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
iAMV.plot()
runcell('AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
iAMV.plot()
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.plot(dat_NA.mean(dim = 'year'),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),d
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
runcell('AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
iAMV.plot()
debugcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
iAMV
G
save_path = os.path.expanduser('~/mtm_local/CESM_LME/obs_SST_HadCRUT5')
dat_glob.to_netcdf(save_path+'obs_SST.nc')
save_path = os.path.expanduser('~/mtm_local/CESM_LME/obs_SST_HadCRUT5/')
dat_NA.to_netcdf(save_path+'obs_SST_NA.nc')
save_path =  os.path.expanduser('~/mtm_local/CESM_LME/')
NA_mask.to_netcdf(save_path+'NA_mask.nc')
OCEANS_mask.to_netcdf(save_path+'OCEANS_mask.nc')
save_path =  os.path.expanduser('~/mtm_local/CESM_LME/masks/')
NA_mask.to_netcdf(save_path+'NA_mask.nc')
OCEANS_mask.to_netcdf(save_path+'OCEANS_mask.nc')
del finalMask_NA,finalMask_OCEANS, temp_mask_OCEANS,temp_mask_NA, OCEANS_mask,
finalMask, ds_mask, OCEANS_shp,OCEANS,NA,shpfile
del finalMask
debugcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
data_resid
data_resid.time.intersection(processed_data.time)
data_resid.year.intersection(processed_data.year)
common_years = ts.year.values[np.where(~np.isnan(ts))[0]]
common_years
data_resid.values
data_resid.values.shape
processed_data.year==data_resid.year
processed_data.where(processed_data.year==data_resid.year)
processed_data.where(processed_data.year==data_resid.year,processed_data.latitude == lat, processed_data.longitude = lon)
processed_data.where(processed_data.year==data_resid.year,processed_data.latitude == lat, processed_data.longitude == lon)
processed_data[latitude == lat]
processed_data[processed_datalatitude == lat]
processed_data[processed_data.latitude == lat]
processed_data.where(latitude == lat)
processed_data.where(processed_data.latitude == lat)
processed_data.where(processed_data.latitude == lat,processed_data.longitude == lon)
processed_data.where(processed_data.latitude == lat,processed_data.longitude == lon).values
processed_data.where(processed_data.year = 1910,processed_data.latitude == lat,processed_data.longitude == lon).values
processed_data.where(processed_data.year == 1910,processed_data.latitude == lat,processed_data.longitude == lon).values
processed_data.all(where(processed_data.year == 1910,processed_data.latitude == lat,processed_data.longitude == lon)).values
arr
ts
arr
ts
arr = xr.DataArray(data = data_filt, dims = ts.dims,coords = {
    'year':year, 'latitude':lat, 'longitude':lon})
arr
data_resid = arr - G
data_resid
xr.combine_by_coords([processed_data,data_resid])
xr.concat([processed_data,data_resid])
xr.concat([processed_data,data_resid],dim = 'year')
processed_data
data_resid
processed_data
processed_data.name
ts
arr = xr.DataArray(data = data_filt, dims = ts.dims,coords = {
    'year':year, 'latitude':lat, 'longitude':lon},name=ts.name)
data_resid = arr - G
xr.combine_by_coords([processed_data,data_resid])
data_resid.name
G
arr
G.values
data_resid
data_resid.name = ts.name
xr.combine_by_coords([processed_data,data_resid])
data_resid
data_resid.dims
ts
ts.dims
arr = xr.DataArray(data = data_filt, dims = processed_data.dims,coords = {
    'year':year, 'latitude':lat, 'longitude':lon})
data_resid = arr - G

data_resid.name = ts.name
processed_data.dims
data_filt
np.expand_dims(data_filt,2)
np.expand_dims(data_filt,1)
np.expand_dims(data_filt,1).shape
np.expand_dims(data_filt,(1,2)).shape
arr = xr.DataArray(data = np.expand_dims(data_filt, (1,2)), dims = processed_data.dims,coords = {
arr = xr.DataArray(data = np.expand_dims(data_filt, (1,2)), dims = processed_data.dims,coords = {
    'year':year, 'latitude':lat, 'longitude':lon})
lat
xr.combine_by_coords([processed_data,data_resid], dim = 'year')
xr.combine_by_coords([processed_data,data_resid])
xr.combine_by_coords([processed_data,data_resid],compat = 'override')
processed_data[1,1,1]
ts.year
year
ts.year==year
processed_data.year
processed_data.year.where(processed_data.year == year)
processed_data.year.equals(year)
temp = processed_data.year.values == year
type(processed_data.year.values)
type(year)
equal_array = np.where((processed_data.year.values[:shorter] == year[:shorter]) & ~np.isnan(processed_data.year.values[:shorter]) & ~np.isnan(year[:shorter]), True, False)
shorter = np.min([processed_data.year.values.size, year.size])
equal_array = np.where((processed_data.year.values[:shorter] == year[:shorter]) & ~np.isnan(processed_data.year.values[:shorter]) & ~np.isnan(year[:shorter]), True, False)
shorter
processed_data.year.values.size
year.size
equal_array.size
temp=processed_data.year.values
temp2=year
temp==temp2
temp==temp2.all()
year
np.where(~np.isnan(ts))[0]
processed_data.year
processed_data.year[np.where(~np.isnan(ts))[0]]
processed_data.where(~np.isnan(ts)[0])
processed_data.where(~np.isnan(ts))
processed_data.where(~np.isnan(ts),1,1)
temp1 = processed_data.sel(latitude = 0,longitude = 0)
temp1 = processed_data.isel(latitude = 0,longitude = 0)
temp1
temp2=data_resid
xr.combine_by_coords([temp1,temp2])
debugcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
np.where(~np.isnan(ts))
np.where(~np.isnan(ts)).shape
np.where(~np.isnan(ts))[0]
debugcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
;lat
lat
lon
debugcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
ds_proc
ds_proc.mean(dim = ['latitude','longitude']).plot()
ds_proc.mean(dim = ['year']).plot()
AMV_corr = xr.corr(ds_proc,iAMV,dim = 'year')
AMV_corr.plot()
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
AMV_corr_p_val = xss.pearson_r_p_value(ds_proc,
                                       iAMV,
                                       dim = 'year',
                                       skipna = False,
                                       keep_attrs = True)
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
AMV_corr_eff_p_val = xss.pearson_r_eff_p_value(ds_proc,
                                       iAMV,
                                       dim = 'year',
                                       skipna = False,
                                       keep_attrs = True)
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_eff_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_eff_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
ds_proc.sel(year = 1850).plot()
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_eff_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(ds_proc.isel(year = 0),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(ds_proc.isel(year = 0),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(ds_proc.isel(year = 0),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(ds.isel(year = 0),cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
ds.isel(year = 0).plot()
ds.isel(year = 100).plot()
ds.isel(year = -1).plot()
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_eff_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
iAMV.plot()
temp= AMV_corr
temp= AMV_corr.values
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =20,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =10,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell('calculate linear regression', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_coef,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =100,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_score,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =10,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_coef,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =10,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_coef,cmap = 'coolwarm',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =10,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_coef,cmap = 'coolwarm',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =50,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_coef,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels = 50,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_reg_score,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels = 50,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels = 50,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
temp=AMV_corr_eff_p_val.values
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr_eff_p_val,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =40,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
sig = 0.05#significance level

fig = plt.figure(figsize=[15,15])

ax = plt.axes(projection=ccrs.Robinson(central_longitude = -90),
                 facecolor='lightgrey')

p = xr.plot.contourf(AMV_corr,cmap = 'bwr',#center = AMV_corr.mean(dim = ['latitude','longitude']),
                  add_colorbar = False,
                  levels =40,
                  transform = ccrs.PlateCarree())
                  # vmin=-0.75, vmax = 0.75)
# coastlines
ax = p.axes.coastlines(color='black')

# grid
gl = p.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xlabel_style = {'size': 18, 'color': 'black'}
gl.ylabel_style = {'size': 18, 'color': 'black'}

# colorbar
cb = plt.colorbar(p, orientation='horizontal',#, pad=0.05,shrink=0.8,
                  ticks = [-0.5,-0.25,0.0,0.25,0.5],
                  label = 'reg coef')


# put dots where p_value is below significance level 
significant_lats, significant_lons = np.where(AMV_corr_eff_p_val < sig) 
significant_lats2, significant_lons2 = np.where(ds.notnull().mean(dim='year') >= 0.95)

for lat, lon in zip(significant_lats, significant_lons):
    actual_lat = AMV_corr_p_val.latitude.values[lat]
    actual_lon = AMV_corr_p_val.longitude.values[lon]
    plt.plot(actual_lon, actual_lat, 'o', color='black', markersize=2, transform=ccrs.Geodetic())
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('plot AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('Gloabal SST anomaly (60S - 60N)', '/Users/afer/.spyder-py3/autosave/reg_index.py')
masks_path = os.expanduser.path('~/mtm_local/CESM_LME/masks')
masks_path = os.path.expanduser('~/mtm_local/CESM_LME/masks')
OCEANS_mask = xr.load_dataarray(masks_path +'OCEANS_mask.nc')
masks_path = os.path.expanduser('~/mtm_local/CESM_LME/masks/')
OCEANS_mask = xr.load_dataarray(masks_path +'OCEANS_mask.nc')
NA_mask = xr.load_dataarray(masks_path +'NA_mask.nc')
runcell('Gloabal SST anomaly (60S - 60N)', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell("process the dataset so only timeseries that don't contain more than 10% NaN values", '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate linear regression', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('plot correlation map', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('plot AMV index', '/Users/afer/.spyder-py3/autosave/reg_index.py')
runcell('calculate correlation between AMV index and processed global temperature data', '/Users/afer/.spyder-py3/autosave/reg_index.py')

## ---(Mon Mar  4 15:12:48 2024)---
runcell(0, '/Users/afer/mtm/MTM-SVD/CESM_LME/mtm/mtm_obs.py')
runcell('Load datasets', '/Users/afer/mtm/MTM-SVD/CESM_LME/mtm/mtm_obs.py')
runcell(0, '/Users/afer/mtm/MTM-SVD/CESM_LME/mtm/mtm_obs.py')
runcell('Load datasets', '/Users/afer/mtm/MTM-SVD/CESM_LME/mtm/mtm_obs.py')