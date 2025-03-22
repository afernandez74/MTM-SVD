import pandas as pd
import matplotlib.pyplot as plt
import os 
import pyleoclim as pyleo
import xarray as xr
pyleo.set_style(style = 'journal',font_scale = 2.0, dpi =300)

#%%
path = os.path.expanduser('~/mtm_local/Gao_etal_2008_data/Gao_08_dat_edit.csv')

dat = pd.read_csv(path)

volc= xr.DataArray(dat["Sulf_Tg"].values, coords={"years": dat["Year"]}, dims="years")

volc_roll = volc.rolling(years=100,center=False).mean()

#%%
fig, ax = plt.subplots(figsize = [25,10])
((volc-volc.mean())/(volc.std())).plot(ax =ax, linewidth = 1,color = 'red')
((volc_roll-volc_roll.mean())/(volc_roll.std())).plot(ax =ax, linewidth = 3, color = 'black')

ax.set_xlim(850,1850)

save_path = os.path.expanduser('~/mtm_local/CESM_LME/figs/Gao_volc_data/')
name = 'glob_volc_inject'

save_fig = input("Save fig? (y/n):").lower()

if save_fig == 'y':
    plt.savefig(save_path+name+'.png', format = 'png', dpi = 300)
    plt.savefig(save_path+name+'.svg', format = 'svg')

else:
    plt.show()
    

#%% wavelet
# series_volc.wavelet().plot()
