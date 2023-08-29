
# Functions for MultiTaper Method-Singular Value Decomposition (MTM-SVD) in python
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
# 
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
# This file contains the functions called in the file
# mmt-svd-python.py
# 
# ------------------------------------------------------------------
# 

from scipy.signal.windows import dpss
from scipy import signal
import numpy as np
from numpy.matlib import repmat


# Function 1) Determine the local fractional variance spectrum LFV 

def mtm_svd_lfv(ts2d,nw,kk,dt,w) :

    # Compute spectrum at each grid point
    p, n = ts2d.shape

    # Remove the mean and divide by std
    vm = np.nanmean(ts2d, axis=0) # mean
    vmrep = repmat(vm,ts2d.shape[0],1)
    ts2d = ts2d - vmrep
    vs = np.nanstd(ts2d, axis=0) # standard deviation
    vsrep = repmat(vs,ts2d.shape[0],1)
    ts2d = np.divide(ts2d,vsrep)
    ts2d = np.nan_to_num(ts2d)
    
    # Apply weights by latitude
    W=w.repeat(p,axis=0)
    ts2d = np.multiply(W,ts2d)
    
    # Slepian tapers
    psi = dpss(p,nw,kk)

    #determine spectral estimation frequencies
    npad = 2**int(np.ceil(np.log2(abs(p)))+2) #frequency range for spectrum domain
        # second nearest power of 2 to the length of the timeseries
    nf = int(npad/2) # fundamental period for spectral estimation
    ddf = 1./(npad*dt) # frequency in npad intervals
    fr = np.arange(0,nf)*ddf #frequency vector
    
    # Get the matrix of spectrums
    psimats = []
    for k in range(kk):
        psimat2 = np.transpose(repmat(psi[k,:],ts2d.shape[1],1) ) 
        psimat2 = np.multiply(psimat2,ts2d)
        psimats.append(psimat2)
    psimats=np.array(psimats)
    nev = np.fft.fft(psimats,n=npad,axis=1)
    nev = np.fft.fftshift(nev,axes=(1)) # temp note: step absent in MATLBA
    nev = nev[:,nf:,:] 

    # Calculate svd for each frequency
    lfvs = np.zeros(nf)*np.nan
    for j in range(nf) :
        U,S,V = np.linalg.svd(nev[:,j,:], full_matrices=False) #svd function
        lfvs[j] = S[0]**2/(np.nansum(S[0:]**2))

    return fr, lfvs



# Function 2) Calculate the confidence interval of the LFV calculations

def mtm_svd_conf(ts2d,nw,kk,dt,niter,sl,w) :

    q = [int(niter*each) for each in sl] # index of 

    partvar = []

    # Bootstrapping
    for it in range(niter):
        print('Iter %i'%it)
        shr = np.random.permutation(ts2d) # random permutation of each time series
        [fr, lfv] = mtm_svd_lfv(shr,nw,kk,dt,w)
        partvar.append(lfv)
    partvar = np.sort(partvar, axis = 0)
    LFVs = partvar
    evalper = []
    for i in q:
        evalper.append(list(partvar[i]))
        
    return fr, evalper, LFVs

def envel(ff0, iif, fr, dt, ddf, p, kk, psi, V) :

    ex = np.ones(p)
    df1 = 0
    c0=1; s0=0;

    c=[c0]
    s=[s0]
    cs=np.cos(2.*np.pi*df1*dt)
    sn=np.sin(2.*np.pi*df1*dt)
    for i in range(1,p) :
        c.append( c[i-1]*cs-s[i-1]*sn )
        s.append( c[i-1]*sn+s[i-1]*cs )
    cl = np.ones(p) ## REMOVE? 
    sl = np.zeros(p) ##

    d = V[0,:]
    d = np.conj(d)*2
    if iif == 1 :
        d = V[0,:] ; d = np.conj(d)

    g = []
    for i0 in range(kk) :
        cn = [complex( psi[i0,i]*c[i], -psi[i0,i]*s[i] ) for i in range(len(s))]
        g.append( ex*cn )
    g=np.array(g).T

    za = np.conj(sum(g))

    [g1,qrsave1] = np.linalg.qr(g)
    dum1 = np.linalg.lstsq( np.conj(qrsave1), np.linalg.lstsq( np.conj(qrsave1.T), d )[0] )[0].T
    amp0=sum(np.conj(za)*dum1)
    dum2 = np.linalg.lstsq( np.conj(qrsave1), np.linalg.lstsq( np.conj(qrsave1.T), za )[0] )[0].T
    amp1=sum(np.conj(za)*dum2)
    amp0=amp0/amp1
    sum1=sum(abs(d)**2)
    d=d-za*amp0
    sum2=sum(abs(d)**2)
    env0= np.linalg.lstsq( np.conj((qrsave1.T)), d.T )[0].T 
    env = np.matmul(g1, env0.T)

    env = env + amp0*np.ones(len(c))

    return env


# Function 3) Reconstruct the spatial patterns associated with peaks in the spectrum

def mtm_svd_recon(ts2d, nw, kk, dt, fo) :

    imode = 0
    lan = 0
    vw = 0

    # Compute spectrum at each grid point
    p, n = ts2d.shape

    # Remove the mean and divide by std
    vm = np.nanmean(ts2d, axis=0) # mean
    vmrep = repmat(vm,ts2d.shape[0],1)
    ts2d = ts2d - vmrep
    vs = np.nanstd(ts2d, axis=0) # standard deviation
    vsrep = repmat(vs,ts2d.shape[0],1)
    ts2d = np.divide(ts2d,vsrep)
    ts2d = np.nan_to_num(ts2d)

    # Slepian tapers
    psi = dpss(p,nw,kk)

    npad = 2**int(np.ceil(np.log2(abs(p)))+2)
    nf = int(npad/2)
    ddf = 1./(npad*dt)
    fr = np.arange(0,nf)*ddf
    
    # Get the matrix of spectrums
    psimats = []
    for k in range(kk):
        psimat2 = np.transpose( repmat(psi[k,:],ts2d.shape[1],1) ) 
        psimat2 = np.multiply(psimat2,ts2d)
        psimats.append(psimat2)
    psimats=np.array(psimats)
    nev = np.fft.fft(psimats,n=npad,axis=1)
    nev = np.fft.fftshift(nev,axes=(1))
    nev = nev[:,nf:,:] 

    # Initialiser les matrices de sorties
    S = np.ones((kk, len(fo)))*np.nan 
    vexp = [] ; totvarexp = [] ; iis = []

    D = vsrep

    envmax = np.zeros(len(fo))*np.nan

    for i1 in range(len(fo)):

        # closest frequency
        iif = (np.abs(fr - fo[i1])).argmin()
        iis.append(iif)
        ff0 = fr[iif]
        print('( %i ) %.2f cyclesyr | %.2f yr'%(iif,ff0,1/ff0))

        U,S0,Vh = np.linalg.svd(nev[:,iif,:].T,full_matrices=False)
        ##V = Vh.T.conj()
        V = Vh
        S[:,i1] = S0

        env1 = envel(ff0, iif, fr, dt, ddf, p, kk, psi, V) # condition 1

        cs=[1]
        sn=[0]
        c=np.cos(2*np.pi*ff0*dt)
        s=np.sin(2*np.pi*ff0*dt)
        for i2 in range(1,p):
            cs.append( cs[i2-1]*c-sn[i2-1]*s )
            sn.append( cs[i2-1]*s+sn[i2-1]*c )
        CS = [complex(cs[i], sn[i]) for i in range(len(cs))]
        CS=np.conj(CS)
    
        # Reconstructions
        R = np.real( D * S[imode, i1] * np.outer(U[:,imode], CS*env1).T )

        vsr=np.var(R,axis=0)
        vexp.append( vsr/(vs**2)*100 )
        totvarexp.append( np.nansum(vsr)/np.nansum(vs**2)*100 )

    return vexp, totvarexp, iis

 
# function 4) calculate annual means from monthly data and reshape 3d data to 2d

# years is the 'year' value of the date field from the .nc file. If it's monthly 
# simulations, the 'years' array will have 12 entries of with the same value
def annual_means(tas,years):
    # obtain array of years values
    years_unique = np.unique(years)
    
    # reshape tas matrix to 2d matrix (time x space)
    tas_ts = tas.reshape((tas.shape[0],tas.shape[1]*tas.shape[2]), order='F')
    
    # calculate annual averages from monthly data
    tas_ts_annual = np.empty((years[-1]-years[0]+1,tas_ts.shape[1]))
    j=0
    for i in years_unique:     
        temp = np.nanmean(tas_ts[years==i,:], axis = 0)
 #           print([i*12, (i*12)+12])
        tas_ts_annual[j,:] = temp
        j=j+1
    return tas_ts_annual

# Function 5) calculate annual means for monthly data and KEEP gridded data format (3d)
# always gets rid of first and last year of data to account for incompleteness
def annual_means_3d(tas,time):
    
    time_steps, spatial_x, spatial_y = tas.shape
    
    # array of all years in time array 
    all_years = np.unique([dt.year for dt in time])
    # array of years whose data will be kept for annual means (all but first and last year)
    years_keep = all_years[1:-1]
        
    # Initialize an array to store yearly means
    yearly_means = np.zeros((len(years_keep), spatial_x, spatial_y))
    
    # mofify 'tas' so that it only contains the years to keep 
    # by obtaining the indexes where the time dimension of 'tas' is within the second 
    # and second-to-last years
    ix = np.where((np.array([dt.year for dt in time]) >= years_keep[0]) & (np.array([dt.year for dt in time]) <= years_keep[-1]))
    tas = tas[ix]
    
    # Calculate annual means
    tas_reshaped = tas.reshape(-1,12,spatial_x,spatial_y)
    tas_annual = np.mean(tas_reshaped, axis=1)
        
    
    return tas_annual, years_keep