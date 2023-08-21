#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:29:13 2023

@author: afer
"""
# functions for the reading and pre-processing of the CESM LME simulation files. 


import xarray as xr
from os import listdir
import pickle as pkl
from mtm_functions_AF import *
import numpy as np
import cftime


# create class "sim" which contains information of each simulation file 
class sim:
    def __init__(self, name, sim_no, tas, time, lat, lon):
        self.name = name
        self.sim_no = sim_no
        self.tas = tas
        self.time = time
        self.lat = lat
        self.lon = lon

def nc_to_dic_CESM(path):
    
    # reads original CESM LME files (26 in total) and saves a python dictionary that
    # contains the temperature data from each file as well as latitude, longitude, simulation years
    # and file name. 
    
    # annual means are NOT calculated

    # data path
    files = listdir(path)
    files.sort()
    
            
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
        
        # find indexes where the simulation number is in the name
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
        sim_lat = ds.lat.values #lat 
        sim_lon = ds.lon.values #lon
        sim_tas = ds.TS.values #tas 3d
        print(i)
        dicti[sim_name] = sim(files_nam[i],sim_no[i],sim_tas,sim_time,sim_lat,sim_lon)
            
    return dicti, sim_no


## ____________________________________________________________________________


def dic_sim_merge_CESM(dicti, sim_no):
    # loads the dictionary created with nc_to_dic_CESM and merges the 
    # monthly values from each of the 13 simulations
    # Returns a dictionary in which each entry is a CESM LME simulation
    
    keys = list(dicti.keys()) # list of keys from old dictionary
    CESM_merged = {}
    lat = dicti[keys[0]].lat
    lon = dicti[keys[0]].lon

    for i in np.unique(sim_no):
    
        
        nam = "".join(["sim",i]) # new key for dictionary w/o dates (eg Sim001)
        print(nam) 
        
        find_str = "".join(['_',i,'_']) # location of the sim number
        
        # keys that contain the desiered simulation in the loop (1-13)
        str_inds = [index for index, s in enumerate(keys) if find_str in s] 
        
        # values of the keys that correspond to the ith simulation
        str_nams = keys[str_inds[0]:str_inds[-1]+1]
        
        tas = np.concatenate([dicti[key].tas for key in str_nams],axis=0)
        time = np.concatenate([dicti[key].time for key in str_nams])
            
        print(i)
        CESM_merged[nam] = sim(nam,i,tas,time,lat,lon)
        
    return CESM_merged
        
## ____________________________________________________________________________

def calc_annual_means_CESM(dicti):
    keys = list(dicti.keys()) # list of keys from old dictionary
    CESM_merged_annual = {} 
    lat = dicti[keys[0]].lat
    lon = dicti[keys[0]].lon
    
    for i_key in keys:
        tas = dicti[i_key].tas
        time = dicti[i_key].time
        tas_annual = annual_means_3d(tas,time)
        CESM_merged_annual[i_key] = sim(dicti[i_key].name,dicti[i_key].sim_no,tas_annual,time_years,lat,lon)
        
    return CESM_merged_annual
    
