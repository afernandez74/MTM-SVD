
from scipy.signal.windows import dpss
from scipy import signal
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt



# Determine the local fractional variance spectrum LFV 

def mtm_svd_lfv(ts2d,nw,kk,dt,w):

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

# Calculate the envelope for the reconstruction of field
def mtm_svd_envel(ff0, iif, fr, dt, ddf, n, k, psi, V):
    
    ex = np.ones(n)

    df1 = 0  # Señal dominante y envolvente son iguales para modos seculares

    c0 = 1
    s0 = 0
    c = np.zeros(n)
    s = np.zeros(n)
    cs = np.cos(2 * np.pi * df1 * dt)
    sn = np.sin(2 * np.pi * df1 * dt)

    c[0] = c0
    s[0] = s0

    for i in range(1, n):
        c[i] = c[i - 1] * cs - s[i - 1] * sn
        s[i] = c[i - 1] * sn + s[i - 1] * cs

    # d = V[:, 0].conj() * 2.0 # complex conjugate of V doubled
    d = V[0,:]
    d = np.conj(d)*2    
    if iif == 1 :
        d = V[0,:] ; d = np.conj(d)

    g = []
    for i0 in range(k) :
        cn = [complex( psi[i0,i]*c[i], -psi[i0,i]*s[i] ) for i in range(len(s))]
        g.append( ex*cn )
    g=np.array(g).T

    za = np.conj(sum(g))

    [g1,qrsave1] = np.linalg.qr(g)

    # Solve for the constant term
    dum1 = np.linalg.lstsq( np.conj(qrsave1).T, np.linalg.lstsq( np.conj(qrsave1.T), d )[0] )[0].T
    amp0=sum(np.conj(za)*dum1)
    dum2 = np.linalg.lstsq( np.conj(qrsave1).T, np.linalg.lstsq( np.conj(qrsave1.T), za )[0] )[0].T
    amp1=sum(np.conj(za)*dum2)
    amp0=amp0/amp1
    sum1=sum(abs(d)**2)
    d=d-za*amp0
    sum2=sum(abs(d)**2)
    env0 = np.linalg.lstsq(np.conj(qrsave1.T), d.conj(), rcond=None)[0].conj()
    env = np.matmul(g1, env0.T)

    env = env + amp0*np.ones(len(c))
    return env


# Signal reconstruction and variance explained for LFV frequency peak chosen
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
    env = mtm_svd_envel(ffo, iif, fr, dt, ddf, n, k, psi, Vh) # condition 1
    
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
    vexp = (vsr/(vs**2))*100
    totvarexp=(np.nansum(vsr)/np.nansum(vs**2))*100

    return R, vsr, vexp, totvarexp, iif




# Calculate the confidence interval of the LFV calculations

def mtm_svd_conf(ts2d,nw,kk,dt,niter,sl,w) :

    q = [int(niter*each) for each in sl] # index of 

    partvar = []
    print("Begin calculation of Confidence Intervals...")
    # Bootstrapping
    for it in range(niter):
        print('Iter %i'%it)
        shr = np.random.permutation(ts2d) # random permutation of each time series
        [fr, lfv] = mtm_svd_lfv(shr,nw,kk,dt,w)
        partvar.append(lfv)
    partvar = np.sort(partvar, axis = 0)
    evalper = []
    for i in q:
        evalper.append(list(partvar[i]))
        
    conflevel = np.asarray(evalper)

    # calculate C.I. mean values for secular and non-secular bands
    fr_sec = nw/(ts2d.shape[0]*dt) # secular frequency value
    fr_sec_ix = np.where(fr < fr_sec)[0][-1] #index in the freq array where the secular frequency is located

    ci_sec = np.nanmean(conflevel[:,0:fr_sec_ix],axis=1) # confidence intervals for secular-and-lower frequencies
    ci_nsec = np.nanmean(conflevel[:,fr_sec_ix+1:],axis=1) # confidence intervals for secular-and-higher frequencies

    ci = np.column_stack((ci_sec,ci_nsec))
    return fr, ci


# Reshape a 3d gridded climate dataset (temp, pressure, etc...) into a 
# 2d array where first dimension is time and the second is space
def reshape_3d_to_2d(mat_3d):
    """
    This function reads in a three-dimensional array of a climate variable that is 
    distributed in latitude, longitude and time
    It then reshapes the matrix so that time is the first dimension and space (latitude-longitude pairs)
    is the second dimension
    """
    if mat_3d.ndim == 3:
        mat_2d = mat_3d.reshape((mat_3d.shape[0],mat_3d.shape[1]*mat_3d.shape[2]), order='F')
    else:
        mat_2d = mat_3d
    return mat_2d

    
# Butterworth low-pass filter
def butter_lowpass(data, cutoff_frequency, sampling_frequency, order=4):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_highpass(data, cutoff_frequency, sampling_frequency, order=4):
    nyquist = 0.5 * sampling_frequency
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

def butter_bandpass(data, low_cutoff_frequency, high_cutoff_frequency, sampling_frequency, order=4):
    nyquist = 0.5 * sampling_frequency
    low_normal_cutoff = low_cutoff_frequency / nyquist
    high_normal_cutoff = high_cutoff_frequency / nyquist
    b, a = signal.butter(order, [low_normal_cutoff, high_normal_cutoff], btype='band', analog=False)
    y = signal.filtfilt(b, a, data)
    return y
    