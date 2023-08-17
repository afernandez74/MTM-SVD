#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:29:13 2023

@author: afer
"""
# reads original CESM LME files (26 in total) and saves a python dictionary that
# contains the temperature data from each file as well as latitude, longitude, simulation years
# and file name. 

# annual means are calculated as well

from mtm_functions_AF import *
import xarray as xr
from os import listdir
import pickle as pkl

# data path
path = "D:\\Work\\MannSteinman_Proj\\Data\\2021_CESM_LME_ALL_FORCING\\2021_CESM_LME_ALL_FORCING\\"
files = listdir(path)
files.sort()

# create class "sim" which contains information of each simulation file 
class sim:
    def __init__(self, name, sim_no, tas, time, lat, lon):
        self.name = name
        self.sim_no = sim_no
        self.tas = tas
        self.time = time
        self.lat = lat
        self.lon = lon
        
# string identifiers for obaining simulation number (out of 13)
id1 = ".f19_g16."
id2 = ".cam."
id3 = ".TS."
id4 = ".nc"

# read the name of each file and the sim number
files_nam = [''] * len(files)
sim_no = [''] * len(files)
sim_yrs = [''] * len(files)
dicti = {}

# loop through the files
for i in range(0, len(files)):
    # read name of simulation
    files_nam[i] = files[i]
    ix1 = files_nam[i].find(id1)
    ix2 = files_nam[i].find(id2)
    
    # simulation number (out of 13)
    sim_no[i] = files_nam[i][ix1+len(id1):ix2]
    
    # indexes for namiming file with simulation dates
    ix3 = files_nam[i].find(id3)
    ix4 = files_nam[i].find(id4)
    sim_yrs[i] = files_nam[i][ix3+len(id3):ix4]
    
    # read data from .nc files 
    ds = xr.open_mfdataset(path+files_nam[i]) # open .nc file
    sim_name = "".join(["sim_",sim_no[i],"_",sim_yrs[i]])# get name
    sim_time = ds.indexes['time'] #get time of sim
    years = sim_time.year # years array 
    sim_lat = ds.lat.values #lat 
    sim_lon = ds.lon.values #lon
    sim_tas = ds.TS.values #tas 3d
    sim_tas = annual_means(sim_tas,years) # calulate annual means from monthly data
    #years = np.unique(years) # unique year values 
    print(i)
    dicti[sim_name] = sim(files_nam[i],sim_no[i],sim_tas,years,sim_lat,sim_lon)
        
dicti_name = "data_merge\\CESM_LME_tas_dict"
 
with open(dicti_name, 'wb') as handle:
    pkl.dump(dicti, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(files_nam, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(sim_no, handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(files_nam, handle, protocol=pkl.HIGHEST_PROTOCOL)

# path to the dictionary
# path_dicti = "D:\\Work\\MannSteinman_Proj\\Code_Python\\MTM_SVD_Python\\mtm_svd_python_master\\CESM_LME_mtm_svd_analysis\\data_merge\\"

## to read the dictionary:
# with open(path_dicti +'CESM_LME_tas_dict', 'rb') as handle:
#    dicti = pkl.load(handle)


    
    

