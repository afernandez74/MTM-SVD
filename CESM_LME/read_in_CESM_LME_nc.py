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
from mtm_functions_AF import annual_means
import numpy as np


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
    
    # annual means are calculated as well

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
        years = sim_time.year # years array 
        sim_lat = ds.lat.values #lat 
        sim_lon = ds.lon.values #lon
        sim_tas = ds.TS.values #tas 3d
        
        ## NOTE NOTE NOTE add function that calculates annual means, only in 3d instead of 2d
        print(i)
        dicti[sim_name] = sim(files_nam[i],sim_no[i],sim_tas,years,sim_lat,sim_lon)
            
    # dicti_name = "data_merge//CESM_LME_tas_dict"
     
    # with open(dicti_name, 'wb') as handle:
    #     pkl.dump(dicti, handle, protocol=pkl.HIGHEST_PROTOCOL)
    #     pkl.dump(files_nam, handle, protocol=pkl.HIGHEST_PROTOCOL)
    #     pkl.dump(sim_no, handle, protocol=pkl.HIGHEST_PROTOCOL)
    #     pkl.dump(files_nam, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return dicti, sim_no


## ____________________________________________________________________________


def dic_sim_merge_CESM(dicti, sim_no):
    # loads the dictionary created with nc_to_dic_CESM and merges the 
    # temperature values from each of the 13 simulations
    
    # Returns a dictionary in which each entry is a CESM LME simulation
     
    # NOTE: only works if there's only two files per simulation! (like in CESM LME files)
    LME_merged = {}
    for i in np.unique(sim_no):
    
        nam = "".join(["sim",i]) # new key for dictionary w/o dates (eg Sim001)
        print(nam) 
        
        keys = list(dicti.keys()) # list of keys from old dictionary
        find_str = "".join(['_',i,'_']) # location of the sim number
        
        # keys that contain the desiered simulation in the loop (1-13)
        str_inds = [index for index, s in enumerate(keys) if find_str in s] 
        
        # values of the keys that correspond to the ith simulation
        str_nams = keys[str_inds[0]:str_inds[-1]+1]
        
        # dictionary entries that correspond to the ith simulation
        dict_1 = dicti[str_nams[0]]
        dict_2 = dicti[str_nams[-1]]
        
        # overlap year
        ol_yr_1 = dict_1.time[-1]
        ol_yr_2 = dict_2.time[0]
        if ol_yr_1==ol_yr_2:
            # weights for averaging 
            w = [(np.count_nonzero(dict_1.time==ol_yr_1))/12, (12-np.count_nonzero(dict_1.time==ol_yr_1))/12]
        
            new_mean = np.average(np.array([dict_1.tas[-1,:],dict_2.tas[0,:]]),axis=0,weights=w)
        
        tas_1 = np.delete(dict_1.tas,(-1),axis=0)
        tas_2 = np.delete(dict_2.tas,(0),axis=0)
        new_sim_tas = np.concatenate((tas_1,new_mean.reshape((1,new_mean.size)),tas_2),axis=0) # new, merged tas matrix. 
        
        time_1 = np.delete(np.unique(dict_1.time),(-1))
        time_2 = np.delete(np.unique(dict_2.time),(0))    
        new_sim_time = np.concatenate((time_1,np.resize(ol_yr_1, 1),time_2)) #new, merged time matrix (years)
        
        print(i)
        LME_merged[nam] = sim(nam,i,new_sim_tas,new_sim_time,dict_1.lat,dict_1.lon)
        
    return LME_merged
        
    
