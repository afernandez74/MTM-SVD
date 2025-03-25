#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 07:41:24 2023

@author: afer
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.path as mpath
import os
import xarray as xr 
from scipy.stats import zscore
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Set global settings for publication-style plots
plt.rcParams.update({
    'font.size': 14,
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'figure.figsize': (8, 4),  # Customize figure size for publication
    'figure.dpi': 300,  # Higher DPI for publication quality
    'lines.linewidth': 2,  # Line width
    'lines.color': 'black',  # Default line color
    'axes.grid': True,  # Show grid
    'axes.titlesize': 16,  # Axis title font size
    'axes.labelsize': 14,  # Axis label font size
    'xtick.labelsize': 12,  # X-tick label font size
    'ytick.labelsize': 12,  # Y-tick label font size
    'xtick.major.size': 6,  # Major tick size for x-axis
    'ytick.major.size': 6,  # Major tick size for y-axis
    'legend.fontsize': 12,  # Legend font size
    'legend.loc': 'best',  # Best location for the legend
    'legend.borderpad': 1.0,  # Padding around the legend
    'grid.linestyle': '--',  # Dashed grid lines
    'grid.color': 'gray',  # Grid line color
    'grid.alpha': 0.5,  # Transparency of grid lines
    'xtick.direction': 'in',  # Tick direction
    'ytick.direction': 'in',  # Tick direction
    'axes.labelpad': 10,  # Padding between axis labels and plot
})

#%% load pages2k datasets
start_year = 1600
AMV_records_only = True
path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets/datasets{start_year}_1850/")
files = os.listdir(path)
files = [entry for entry in files if not entry.startswith('.')]
files = sorted(files)

# concatenate them into single dataset
# ds_all = xr.open_mfdataset(f'{path}/*.nc', combine='nested', concat_dim = 'record')

#%% organize data into archive types for plotting

records = []

# create dataframe with all metadata of all records
archive_types = []

for file in files:
    
    ds_i = xr.open_dataset(path+file)
    
    # Extract relevant attributes
    record = {
        "proxy_type": ds_i.attrs.get("proxy_type", None),
        "units": ds_i.attrs.get("units", None),
        "site_name": ds_i.attrs.get("site_name", None),
        "variable_name": ds_i.attrs.get("variable_name", None),
        "archive_type": ds_i.attrs.get("archive_type", None),
        "data_set_name": ds_i.attrs.get("data_set_name", None),
        "pages2kID": ds_i.attrs.get("pages2kID", None),
        "lat": ds_i.attrs.get("lat", None),
        "lon": ds_i.attrs.get("lon", None),
        "elev": ds_i.attrs.get("elev", None),
        "direction": ds_i.attrs.get("direction", None),
        "AMV": ds_i.attrs.get("AMV",None)
    }
    records.append(record)

archives = pd.DataFrame(records)
archive_types = set(archives.archive_type)
if AMV_records_only:
    archives = archives.loc[archives['AMV']=='Yes']
#%%
# read data, normalize and assign to archive_dat dictionary 
# archive_dat keys-> archive types and values -> lists with all datasetsn normalized

archive_dat = {}
all_archs = []
for archive_type in archive_types:
    archive_dat[archive_type]=[]    
    for file in files:
        ds_i = xr.open_dataset(path+file)
        AMV_i = ds_i.AMV
        if ds_i.archive_type == archive_type:
            attrs = ds_i.attrs
            if attrs['direction'] == 'positive':
                ds_i = (ds_i - ds_i.mean()) / ds_i.std() #standardize
            else: 
                ds_i = -1* (ds_i - ds_i.mean()) / ds_i.std() # standardize and flip
            ds_i.attrs = attrs
            if not AMV_records_only:
                archive_dat[archive_type].append(ds_i) #append to appropriate archive entry
                all_archs.append(ds_i)
            elif AMV_i:
                archive_dat[archive_type].append(ds_i) #append to appropriate archive entry
                all_archs.append(ds_i)
            
#%%combine lists in dat_dic for easier handling of data and plotting

dat_dic = {}
for archive_type, list_i in archive_dat.items():
    print(archive_type)
    ds_list= []
    for arch in list_i:
        if archive_type == 'marine sediment':
            print(arch)
        var_name = arch.archive_type + '_' + arch.pages2kID
        var_name = var_name.replace(' ','_') if '' in var_name else var_name
        # change name of variable so the data is concatenateable
        arch_rename = arch.rename({"proxy_data": var_name})
        ds_list.append(arch_rename)
    ds_new = xr.merge(ds_list)
    dat_dic[archive_type] = ds_new
        
#%%plot timeseries mean and standard deviations

for archive_type,ds in dat_dic.items():
    mean_ts = ds.to_array(dim='arch').mean(dim='arch')
    std_ts = ds.to_array(dim='arch').std(dim='arch')
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.5,0.8])
    ax.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
    ax.grid(True,which='major',axis='both')


    p1 = mean_ts.plot(ax=ax,
            linestyle = '-',
            linewidth=2,
            zorder = 10,
            color = 'black',
            label = f'mean {archive_type}')

    # plot +1sd and -1sd shaded areas
    p3 = ax.fill_between(std_ts.time.values, mean_ts+std_ts, mean_ts-std_ts,
                    alpha=.5, linewidth=0.1, zorder = 2, color = 'lightgray',
                    label = '\u00B1 \u03C3')
    plt.title(f'{archive_type}_N={len(ds.data_vars)}')
    plt.xlabel('Year AD')
    plt.ylabel('Z-score proxy')

#%%
# Predefine the styles for each archive_type (color and marker and zorder)
style_dict = {
    'tree': ('green','^',1),
    'borehole': ('orange', 'v', 5),      
    'coral': ('red','h', 5),      
    'documents': ('blue','s',3),   
    'glacier ice': ('slategrey','o',3),  
    'hybrid': ('indigo','X',5),  
    'lake sediment': ('brown','d',3),   
    'marine sediment': ('turquoise','*',3),
    'sclerosponge': ('olive','<',5),
    'speleothem': ('magenta','p',3),
}
# how many of each type there is 
counts = archives.groupby('archive_type').size().reset_index(name='count')

# Create the figure and map

fig_size = (10,10) if AMV_records_only else (10, 5) 
fig = plt.figure(figsize=fig_size)

projection = ccrs.AlbersEqualArea(central_longitude=-40) if AMV_records_only else ccrs.Robinson(central_longitude=-40)

ax = fig.add_subplot(1,1,1, projection=projection, facecolor=None)

if AMV_records_only:
    ax.set_extent([-100,40,0,80],crs=ccrs.PlateCarree())
    vertices = [(lon, 0) for lon in range(-100, 31, 1)] + \
           [(lon, 80) for lon in range(30, -101, -1)]
    boundary = mpath.Path(vertices)
    ax.set_boundary(boundary, transform=ccrs.PlateCarree())

ax.set_global()
ax.coastlines(zorder=1)
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gainsboro')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':')

ax.gridlines(linewidth=0.5,zorder=0.9)
# Plot data points using the predefined styles
for _, row in archives.iterrows():
    archive_type = row['archive_type']
    color,marker,z = style_dict[archive_type]
    count = counts[counts.archive_type==archive_type]['count'].values[0]
    ax.scatter(float(row['lon']), float(row['lat']),
               color=color, 
               marker=marker, 
               transform=ccrs.PlateCarree(),  # Transform lat/lon to map projection
               s=10,  # Marker size
               label=f'{archive_type}, N={count}',
               zorder=z
                   )
# Add a legend to the plot
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # Remove duplicates if needed
ax.legend(by_label.values(), by_label.keys(), 
          loc='lower right', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12)
plt.title(f'Compiled archives for {start_year}-1850CE \nN={counts.sum(axis=0)[1]}')
plt.show()
#%% indiv plots 
# for arch in all_archs:
#     fig=plt.subplots()
#     arch.proxy_data.plot()
#     plt.title(arch.data_set_name)
#     plt.savefig(
#         os.path.expanduser(f'/Users/afer/mtm_local/pages2k/indiv_ts_plots{start_year}/')+
#         arch.data_set_name+'.png',
#         dpi=100)
#     plt.close()
