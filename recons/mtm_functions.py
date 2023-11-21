# Script for MultiTaper Method-Singular Value Decomposition (MTM-SVD) with Monte Carlo test in python

# ------------------------------------------------------------------

# This script is a modified version of the Python function developed by
# Mathilde Jutras at McGill University, Canada[1]. 
# You can find the original Python code here: 
# https://github.com/mathildejutras/mtm-svd-python

# This script was adapted by Yitao Liu at Nanjing University of Information Science & Technology, China
# Copyright (C) 2021, Yitao Liu
# and is available under the GNU General Public License v3.0

# The script may be used, copied, or redistributed as long as it is cited as follow:
# Liu-Yitao, & Mathilde Jutras. (2021). Liu-Yitao/mtm-svd-python: MTM-SVD with Monte Carlo test (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.5774584

# This software may be used, copied, or redistributed as long as it is not 
# sold and that this copyright notice is reproduced on each copy made. 
# This routine is provided as is without any express or implied warranties.

# Questions or comments to:
# Yitao Liu, liuyitao97@outlook.com

# Last update:
# Dec 2021

# ------------------------------------------------------------------

# The script is structured as follows:

# In the main script is found in mtm-svd-python.py
# In the first section, the user can load the data,
# assuming the outputs are stored in a netcdf format.
# In the secton section, functions are called to calculate the spectrum
# The user will then be asked for which frequencies he wants to plot 
# the spatial patterns associated with the variability.
# In the third section, the spatial patterns are plotted and saved

# The required functions are found in mtm_functions.py

# ------------------------------------------------------------------

# Python Package needed:
# - numpy
# - scipy
# - xarray (read the netcdf file)
# - matplotlib (not necessary, just for plotting)

# You can install the needed Python packages by conda,with the command below
# ```
# conda install -c conda-forge numpy scipy xarray matplotlib
# ```

# [1] Mathilde Jutras. (2020, July 6). mathildejutras/mtm-svd-python: v1.0.0-alpha (Version v1.0.0). Zenodo. http://doi.org/10.5281/zenodo.3932319

# ------------------------------------------------------------------

from scipy.signal.windows import dpss
from scipy import signal
import numpy as np
from numpy.matlib import repmat
from numpy.random import shuffle
# from numba import jit

# Function 1) Determine the local fractional variance spectrum LFV 
# @jit
def mtm_svd_lfv(ts2d,nw,kk,dt) :

	# Compute spectrum at each grid point
	p, n = ts2d.shape

	# Remove the mean and divide by std
	# axis_t = np.asarray([0,]).astype(np.int32)
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
		psimat2 = np.transpose(repmat(psi[k,:],ts2d.shape[1],1) ) 
		psimat2 = np.multiply(psimat2,ts2d)
		psimats.append(psimat2)
	psimats=np.array(psimats)
	nev = np.fft.fft(psimats,n=npad,axis=1)
	nev = np.fft.fftshift(nev,axes=(1))
	nev = nev[:,nf:,:] 

	# Calculate svd for each frequency
	lfvs = np.zeros(nf)*np.nan
	for j in range(nf) :
		U,S,V = np.linalg.svd(nev[:,j,:], full_matrices=False)
		lfvs[j] = S[0]**2/(np.nansum(S[1:])**2)

	

	return fr, lfvs



# Function 2) Calculate the confidence interval of the LFV calculations
# @jit
def monte_carlo_test(index_with_space,niter,sl,len_freq,nw,kk,dt):
    '''
    sample:
        monte_carlo_test(index_with_space,niter,sl,len(freq),nw,kk,dt)
    '''

    # create file to store all random lfv
    lfv_mc = np.zeros((niter, len_freq))
    # calculate all random lfc and store in lfv_mc[num_of_mc, num_of_freq]
    for ii in range(niter):
        if (ii % 10) == 0:
            print(f'niter = {ii}')
        index_with_space_rd = index_with_space.copy()
        shuffle(index_with_space)
        [freq_rd, lfv_rd] = mtm_svd_lfv(index_with_space,nw,kk,dt)
        lfv_mc[ii,:] = lfv_rd
    lfv_mc_sort = np.sort(lfv_mc, axis=0)# true?
    
    num = np.rint((1-np.asarray(sl)) * niter).astype(np.int64)-1
    # print(lfv_mc_sort.shape)
    # print(num)
    conflev = lfv_mc_sort[num,:]
    
    return freq_rd,conflev


# @jit
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
# @jit
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
