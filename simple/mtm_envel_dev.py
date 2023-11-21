import numpy as np

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
    d = V[0,:].conj() * 2.0 # complex conjugate of V doubled
    # multiply the tapers by the sinusoid to get kernels
    # note we use the complex conjugate
    g = np.zeros((k,n))
    for i0 in range(k):
        g[i0,:] = ex * (psi[i0,:] * c - 1j * psi[i0,:] * s)
    g = np.array(g).T
    za = np.conj(sum(g))

    # orthogonal decomposition of the tapers
    [g1,qrsave1] = np.linalg.qr(g, mode = 'complete')

    # Solve for the constant term
    dum1 = np.linalg.solve(qrsave1.conj().T, np.linalg.solve(qrsave1, d)).T
    amp0 = sum(np.conj(za)*dum1)
    dum2 = np.linalg.solve(qrsave1.conj().T, np.linalg.solve(qrsave1, za)).T
    amp1=sum(np.conj(za)*dum2)
    amp0=amp0/amp1
    sum1=sum(abs(d)**2)
    d=d-za*amp0
    sum2=sum(abs(d)**2)
    env0= np.linalg.lstsq( np.conj((qrsave1.T)), d.T )[0].T 
    env = np.matmul(g1, env0.T)

    env = env + amp0*np.ones(len(c))
    return env
