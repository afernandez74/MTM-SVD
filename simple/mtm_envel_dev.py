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
