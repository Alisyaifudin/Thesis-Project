from operator import itemgetter
from scipy.interpolate import interp1d
import numpy as np
from scipy.integrate import odeint

def sech(x):
    return 1/np.cosh(x)

def rhoDD(z, sigmaDD, hDD):
    return sigmaDD/(4*hDD)*sech(z/(2*hDD))**2

def frho(rho0, phi, sigmaz ):
    return rho0*np.exp(-phi/sigmaz**2)

def rho_tot(z, phi, rhos, sigmaz, rhoDM, sigmaDD, hDD, R=3.4E-3):
    rho = np.array(list(map(lambda par: frho(par[0], phi, par[1]), zip(rhos, sigmaz))))
    return rho.sum() + rhoDM + rhoDD(z, sigmaDD, hDD) - R

def f(u, z, rhos, sigmaz, rhoDM, sigmaDD, hDD, R=3.4E-3):
    G = 4.30091E-3 # pc/M_sun (km/s)^2
    return (u[1], 4*np.pi*G*rho_tot(z, u[0], rhos, sigmaz, rhoDM, sigmaDD, hDD, R))

def phi_mod(zz, **theta):
    args = ('rhos', 'sigmaz', 'rhoDM', 'sigmaDD', 'hDD', 'zsun', 'R')
    rhos, sigmaz, rhoDM, sigmaDD, hDD, zsun, R = itemgetter(*args)(theta)
    res = 1000
    if 'res' in theta:
        res = theta['res']
    phi0 = 0 # (km/s)^2
    Kz0 = 0 # pc (km/s)^2

    y0 = [Kz0, phi0]
    zmax = np.max(np.abs(zz+zsun))
    zs = np.linspace(0, zmax, res)
    us = odeint(f, y0, zs, args=(rhos, sigmaz, rhoDM, sigmaDD, hDD, R))
    phi = us[:, 0]
    phi_interp = interp1d(zs, phi, kind='cubic')
    phi_z = lambda z, zsun: phi_interp(np.abs(z+zsun))
    return phi_z(zz, zsun)

def nu_mod(zz, **theta):
    args = ('sigma_w', 'a', 'nu0')
    sigma_w, a, nu0 = itemgetter(*args)(theta)
    sigma_w = np.vstack(sigma_w)
    a_raw = np.vstack(a)
    a = a_raw/np.sum(a_raw)
    nu = nu0*np.sum(a*np.exp(-phi_mod(zz, **theta)/sigma_w**2), axis=0)
    nu = np.abs(nu)
    return nu