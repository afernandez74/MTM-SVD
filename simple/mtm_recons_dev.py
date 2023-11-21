# test 

import numpy as np
from scipy.fftpack import fft
from scipy.signal.windows import dpss
from scipy.interpolate import interp1d
from scipy.linalg import svd
from numpy.matlib import repmat
from mtm_envel_dev import mtm_svd_envel as envel


def mtm_svd_bandrecon(ts2d, nw, k, dt, fo, w):
    
    imode = 0
    n, p = ts2d.shape

    # calculate mean and remove
    vm = np.nanmean(ts2d, axis=0)
    vmrep = repmat(vm,ts2d.shape[0],1)
    ts2d = ts2d - vmrep
    
    # calculate std and remove
    vs = np.nanstd(ts2d, axis=0)
    vsrep = repmat(vs,ts2d.shape[0],1)
    ts2d = np.divide(ts2d,vsrep)
    ts2d = np.nan_to_num(ts2d)
    
    # Apply weights by latitude
    W=w.repeat(n,axis=0)
    ts2d = np.multiply(W,ts2d)
    
    #determine spectral estimation frequencies    
    npad = 2**int(np.ceil(np.log2(abs(n)))+2)  # second nearest power of 2 to the length of the timeseries
    nf = int(npad/2) # fundamental period for spectral estimation
    ddf = 1./(npad*dt) # frequency in npad intervals
    fr = np.arange(0,nf)*ddf #frequency vector

    # Slepian tapers
    psi = dpss(n,nw,k)

    # Get the matrix of spectrums
    nev = []
    psimats = []
    for kk in range(k):
        psimat2 = np.transpose(repmat(psi[kk,:],ts2d.shape[1],1) ) 
        psimat2 = np.multiply(psimat2,ts2d)
        psimats.append(psimat2)
    psimats=np.array(psimats)
    nev = np.fft.fft(psimats,n=npad,axis=1)
    nev = np.fft.fftshift(nev,axes=(1)) 
    nev = nev[:,nf:,:] 

    # define output matrices
    
    R = np.zeros((n, p))
    vexp = [] ; 
    totvarexp = [] ; 
    
    D = vsrep
    
    
    #closest frequency to user-defined value
    iif = np.argmin(np.abs(fr - fo)) #index 
    iif = (np.abs(fr - fo)).argmin()
    ffo = fr[iif] #freq value

    # perform SVD on MTM matrix
    U,S,Vh = np.linalg.svd(nev[:,iif,:].T,full_matrices=False)

    # calculate envelope
    env = envel(ffo, iif, fr, dt, ddf, n, k, psi, Vh) # condition 1
    
    # calculate sinusoids for reconstruction
    cs=[1]
    sn=[0]
    c=np.cos(2*np.pi*ffo*dt)
    s=np.sin(2*np.pi*ffo*dt)
    for i2 in range(1,n):
        cs.append( cs[i2-1]*c-sn[i2-1]*s )
        sn.append( cs[i2-1]*s+sn[i2-1]*c )
    CS = [complex(cs[i], sn[i]) for i in range(len(cs))]
    CS = np.conj(CS)
    
    # Reconstruction
    R = np.real( D * S[imode] * np.outer(U[:,imode], CS*env).T )
    
    # Variance explained
    vsr=np.var(R,axis=0)
    vexp = (vsr/(vs**2)*100)
    totvarexp=(np.nansum(vsr)/np.nansum(vs**2)*100)

    return R, vsr, vexp, totvarexp, iif