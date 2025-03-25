# calculate the LFV all proxies

# %% import functions and packages

from mtmsvd_funcs import mtm_svd_lfv,mtm_svd_conf,quick_lfv_plot,mtm_svd_bandrecon
import os
import xarray as xr
import os 
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,BoundaryNorm
import matplotlib.path as mpath
mpl.rcParams.update(mpl.rcParamsDefault)
# Set global settings for plots
plt.rcParams.update({
    'font.size': 14,
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    # 'figure.figsize': (8, 4),  # Customize figure size for publication
    # 'figure.dpi': 300,  # Higher DPI for publication quality
    # 'lines.linewidth': 2,  # Line width
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
    # 'legend.borderpad': 1.0,  # Padding around the legend
    'grid.linestyle': '--',  # Dashed grid lines
    'grid.color': 'gray',  # Grid line color
    'grid.alpha': 0.5,  # Transparency of grid lines
    'xtick.direction': 'in',  # Tick direction
    'ytick.direction': 'in',  # Tick direction
    'axes.labelpad': 10,  # Padding between axis labels and plot
})


#%% Load proxy data
start_yr = 1600
AMV_records_only = True
path = os.path.expanduser(f"~/mtm_local/proxy_data/compiled_datasets/datasets{start_yr}_1850/")
files = os.listdir(path)
files = [entry for entry in files if not entry.startswith('.')]
files = sorted(files)
#%%
records_mdat = []
records = []
# create dataframe with all metadata of all records
# create list of datasets with all normalized records
for file in files:
    
    ds_i = xr.open_dataset(path+file)
    
    var_name = ds_i.archive_type + '_' + ds_i.pages2kID
    var_name = var_name.replace(' ','_') if '' in var_name else var_name

    # METADATA 
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
        "new_name": var_name, 
        "AMV": ds_i.attrs.get("AMV",None)
    }
    records_mdat.append(record)
    AMV = ds_i.AMV
    # DATA
    attrs = ds_i.attrs
    if attrs['direction'] == 'positive':
        ds_i = (ds_i - ds_i.mean()) / ds_i.std() #standardize
    else: 
        ds_i = -1* (ds_i - ds_i.mean()) / ds_i.std() # standardize and flip
            
    # change name of variable so the data is concatenateable
    ds_i_rename = ds_i.rename({"proxy_data": var_name})
    if not AMV_records_only:
        records.append(ds_i_rename)
    elif AMV == 'Yes':
        records.append(ds_i_rename)
records = xr.merge(records)

records_mdat = pd.DataFrame(records_mdat)

if AMV_records_only:
    records_mdat=records_mdat.loc[records_mdat['AMV']=='Yes']

#%% MTM_SVD analysis 
# parameters for mtm-svd analysis
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

# parameters for confidence interval estimation
niter = 100    # Recommended -> 1000
sl = [.99,.95,.9,.8,.5] # confidence levels

chunk = False # Analyze a specific timperiod of the data

#if chunk, define the period of analysis
year_i = 1150
year_f = 1400

tas = records.to_array(dim='arch')
records_lats = records_mdat['lat'].values
records_lons = records_mdat['lon'].values


LFV = []

# temp data
if chunk:
    tas = ds_i.sel(time=slice(year_i,year_f))

tas_np = tas.to_numpy() 

tas_2d = np.transpose(np.array(tas_np)) # rows are time and columns space

w = np.ones((1,tas_2d.shape[1]))

# calculate the LFV spectrum
freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)

# assign results to xr dataarray
spectrum = xr.DataArray(
    data = lfv,
    dims = 'freq',
    coords = [freq],
    name = f'{year_i}_{year_f}_lfv' if chunk else f'{start_yr}_1850_lfv'
    )

LFV.append(spectrum)
[conffreq, conflevels] = mtm_svd_conf(tas_2d,nw,kk,dt,niter,sl,w)
ci=conflevels[:,1]

fr_sec = nw/(tas_2d.shape[0]*dt) # secular frequency value
T_sec = 1/fr_sec
fr_sec_ix = np.where(conffreq < fr_sec)[0][-1] #index in the freq array where the secular frequency is located

# quick_lfv_plot(spectrum, ci)

#%% variance explained (loadings) map

#locate maxima in spectrum
fr_max = spectrum.sel(freq=slice(1/80,1/40)).idxmax()

fo = fr_max.values

# fo = 0.02148438

R, vsr, vexp, totvarexp, iif = mtm_svd_bandrecon(tas_2d,nw,kk,dt,fo,w)
RV = np.transpose(np.array(R)) # flip columns and rows again

RV = xr.DataArray(
    data = RV,
    dims = tas.dims,
    coords = tas.coords,
    name = f'Var exp{1./fo:.2f} yr period'
)

VEXP = xr.DataArray(
    data = vexp,
    dims = 'arch',
    coords = {'arch': list(records.keys())},
    name = f'Var exp{1./fo:.2f} yr period'
)

VEXP_points = xr.DataArray(
    VEXP.values, 
    dims=['point'], 
    coords={'lat': ('point', records_lats), 'lon': ('point', records_lons)}
)

print(f'total variance explained by {(1/fo):.2f} = {totvarexp:.2}')

# =============================================================================
# plot spectrum
# =============================================================================
fig,ax1 = plt.subplots(figsize=(8,4))

xticks = [100, 80, 60, 40, 30, 20]
ax1.set_xscale('log')
xticks2 = [1/x for x in xticks]
ax1.set_xticks(xticks2)
ax1.set_xticklabels([str(x) for x in xticks])
ax1.grid(True, which='major', axis='both')
ax1.set_xlim((xticks2[0], xticks2[-1]))
ax1.set_ylim(0.3, 0.8)
ax1.tick_params(axis='x', which='major', direction='out', pad=15, labelrotation=45)
ax1.minorticks_off()

p1 = spectrum.plot(linestyle='-', linewidth=2, zorder=10, color='blue', ax=ax1)

[ax1.axhline(y=i, color='black', linestyle='--', alpha=0.8, zorder=1) for i in ci]
ax1.plot(freq[iif], lfv[iif], 'r*', markersize=20, zorder=20)
ax1.legend()
if AMV_records_only: 
    ax1.set_title(f'AMV proxy {start_yr}-1850 LFV')
else:    
    ax1.set_title(f'All proxy {start_yr}-1850 LFV')

ax1.set_xlabel('LFV')
ax1.set_ylabel('Period (yr)')

# =============================================================================
# map
# =============================================================================

projection = ccrs.AlbersEqualArea(central_longitude=-40) if AMV_records_only else ccrs.Robinson(central_longitude=-40)
fig_size = (12,12) if AMV_records_only else (9,5) 

fig = plt.figure(figsize=fig_size)


ax = plt.axes(projection=projection)

ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='gainsboro')


# cmap = plt.cm.hot_r  # You can choose any colormap you prefer
# norm = BoundaryNorm(boundaries, cmap.N)

sc = ax.scatter(VEXP_points['lon'], VEXP_points['lat'], 
                 c=VEXP_points.values, 
                 cmap='hot_r', 
                 edgecolor='black', s=100, 
                 transform=ccrs.PlateCarree(),
                 # norm = LogNorm(vmin=1,vmax=VEXP.quantile(0.9))
                 vmin=0,
                 vmax = VEXP.quantile(0.95)                 
                 )
if AMV_records_only:
    ax.set_extent([-100,40,0,80],crs=ccrs.PlateCarree())
    vertices = [(lon, 0) for lon in range(-100, 31, 1)] + \
           [(lon, 80) for lon in range(30, -101, -1)]
    boundary = mpath.Path(vertices)
    ax.set_boundary(boundary, transform=ccrs.PlateCarree())
ax.set_global()

if AMV_records_only: 
    ax.set_title(f'AMV proxy {start_yr}-1850 var exp\nVariance explained {1./fo:.2f} yrs = {totvarexp:.2f}%')
else:    
    ax.set_title(f'All proxy {start_yr}-1850 var exp\nVariance explained {1./fo:.2f} yrs = {totvarexp:.2f}%')


# Add separate colorbar
cb = plt.colorbar(sc, orientation='horizontal', label='% Variance Explained')

# Gridlines
gl = sc.axes.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                       linewidth=1, color='gray', alpha=0.5, linestyle='--')

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()


# # merge all dataArrays into a single dataset|
# lfv = xr.Dataset(LFV)
# lfv.attrs["period"] = f'{year_i}-{year_f}'

# # name file to save
# name = 'lfv'
# name = name + f'_{year_i:04}-{year_f:04}'if chunk else name
# name = name + '_NA' if NA else name
# name = name + '_unforc' if unforced else name

# #save results dataset as .nc file
# path = os.path.expanduser('~/mtm_local/CESM_LME/mtm_svd_results/lfv/')
# lfv.to_netcdf(path + name +'.nc')

