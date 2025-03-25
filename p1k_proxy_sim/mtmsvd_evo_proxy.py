
# calculate the LFV spectra of all members of the CESM LME ensemble

# %% import functions and packages

import xarray as xr
import os 
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
from mtmsvd_funcs import mtm_svd_lfv,mtm_svd_conf,quick_lfv_plot,mtm_svd_bandrecon
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,BoundaryNorm
# Set global settings for plots
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
start_yr = 1400
AMV_records_only = False

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



#%% Evolving LFV spectra 
# parameters for mtm-svd analysis
nw = 2; # bandwidth parameter
kk = 3; # number of orthogonal windows
dt = 1 # annual data (12 if monthly)

# # parameters for confidence interval estimation
# niter = 1000    # Recommended -> 1000
# sl = [.99,.95,.9,.8,.5] # confidence levels

tas = records.to_array(dim='arch')
records_lats = records_mdat['lat'].values
records_lons = records_mdat['lon'].values

wndw = 200 # window of MTM-SVD analysis 

rez = 5 # window moves every 'rez' years

wndw2 = np.floor(wndw/2) #half window

year_i = int(tas.time[0].values + np.floor(wndw/2))
year_f = int(tas.time[-1].values - np.floor(wndw/2))

LFV = []

yrs=[]
spectra=[]

for yr in range (year_i,year_f,rez):
    
    range_yrs = slice(yr - wndw2, yr + wndw2-1)
    print(range_yrs)
    tas_temp = tas.sel(time = range_yrs)
    tas_np = tas_temp.to_numpy() 

    tas_2d = np.transpose(np.array(tas_np)) # rows are time and columns space
    w = np.ones((1,tas_2d.shape[1]))

    freq,lfv = mtm_svd_lfv(tas_2d, nw, kk, dt, w)
    
    spectrum = xr.DataArray(
        data = lfv,
        dims = ['freq'],
        coords = dict(freq = freq),
        )
    yrs.append(yr)
    spectra.append(spectrum)

LFV = xr.concat(spectra, dim='mid_yr').assign_coords(mid_yr=yrs)

#%% plot evo spectrum

whole_spec = False
cmap = 'inferno'

fig = plt.figure(figsize=(25,10))

ax1 = plt.subplot(121)

ax1.set_yscale('log')

ax1.minorticks_off()

# pick which periods to showcase (years)
yticks = [100,90,80,70,60,50,40,30,20]
if whole_spec:
    yticks = [100,80,70,60,50,40,30,20,10,5,3]

yticks2 = [1/x for x in yticks]
ax1.set_yticks(yticks2)
yticks2_labels = [str(x) for x in yticks]
ax1.set_yticklabels(yticks2_labels)
ax1.tick_params(axis='x',which='major',direction='out',
               pad=15, labelrotation=45)
ax1.set_xlabel('year')
ax1.set_ylabel('Period [yrs]')


LFV.plot.contourf(
    ax = ax1,
    add_labels = False,
    x='mid_yr',
    y='freq',
    ylim = [1/yticks[0],1/yticks[-1]],
    # robust = True,
    vmin = 0.4,
    # vmax = 0.8 ,
    cmap = cmap,
    levels = 15
    )

title1 = f'lfv evo AMV proxies {start_yr}-1850' if AMV_records_only else f'lfv evo all proxies {start_yr}-1850'
ax1.set_title(title1)
plt.show()


# #%%
# # merge all dataArrays into a single dataset
# lfv = xr.Dataset(LFV)

# #save results dataset as .nc file
# path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv_evo/")
# name = f'lfv_evo_{rez}yr_{wndw}window'

# name  = name + '_EM' if EM else name
# if not EM:
#     if unforced:
#         name= name + '_unforced'
#     else:
#         name = name + '_forc'

# if NA:
#     name = name + '_NA'

# lfv.to_netcdf(path + name + '.nc')

# %% Confidence intervals 

# niter = 1000    # Recommended -> 1000
# sl = [.99,.95,.9,.8,.5] # confidence levels
# wndw = 200 # window of MTM-SVD analysis 

# case = 'ALL_FORCING'
# run = 6
# unforced = False
# NA = False
# ctr_yr = 1400 #center year for confidence interval calculation


# wndw2 = int(np.floor(wndw/2)) #half window
# range_yrs = slice(ctr_yr - wndw2, ctr_yr + wndw2-1)

# # ref sim
# dat = CESM_LME_unforced if unforced else \
#     CESM_LME

# tas_ref = dat[case].isel(run = run).sel(year=slice(ctr_yr-wndw2,ctr_yr+wndw2)) if not NA else \
#     dat[case].isel(run = run).where(NA_mask == 1).sel(year=slice(ctr_yr-wndw2,ctr_yr+wndw2))

# tas_ref_np = tas_ref.TS.to_numpy()

# tas_ref_2d = reshape_3d_to_2d(tas_ref_np)

# # Weights based on latitude
# [xx,yy] = np.meshgrid(tas_ref.lon,tas_ref.lat)
# w = np.sqrt(np.cos(np.radians(yy)));
# w = w.reshape(1,w.shape[0]*w.shape[1],order='F')

# # conflevels -> 1st column secular, 2nd column non secular 
# print(f'Confidence Interval Calculation for ref sim ({niter} iterations)...')
# [conffreq, conflevels] = mtm_svd_conf(tas_ref_2d,nw,kk,dt,niter,sl,w)

# #save results
# path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv_evo/")

# name = f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_unforced_NA.npy' if unforced and NA else \
#         f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_unforced.npy' if unforced and not NA else \
#         f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}_NA.npy' if NA and not unforced else \
#         f'conf_int_{case}_run{run}_yrs{ctr_yr-wndw2}-{ctr_yr+wndw2}.npy' 
        
# np.save(path+name ,conflevels)

# # =============================================================================
# # Rescaling of confidence intervals 
# # =============================================================================

# fr_sec = nw/(tas_ref_2d.shape[0]*dt) # secular frequency value
# fr_sec_ix = np.where(conffreq < fr_sec)[0][-1] 

# # load CNTL lfv 
# path = os.path.expanduser("~/mtm_local/CESM_LME/mtm_svd_results/lfv/")
# lfv_ref = xr.open_dataset(path + 'CNTL.nc')

# #calculate mean for non-secular band only 
# lfv_mean = lfv_ref.isel(freq=slice(fr_sec_ix,-1)).mean()
# lfv_mean = np.nanmean(lfv_ref[fr_sec_ix:]) # mean of lfv spectrum in the nonsecular band 
# mean_ci = conflevels[-1,-1] # 50% confidence interval array (non secular)

# adj_factor = lfv_mean/mean_ci # adjustment factor for confidence intervals
# adj_ci = conflevels * adj_factor # adjustment for confidence interval values

# np.save(path+'conf_int_CNTL.npy',conflevels)

