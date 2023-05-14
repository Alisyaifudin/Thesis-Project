import numpy as np
from scipy.stats import norm
from .gravity import phi_mod

def n_gaussian_phi(w, phi, *popt):
    n = len(popt) // 3
    y = 0
    for i in range(n):
        a, mu, sigma = popt[3*i:3*i+3]
        ws = w - mu
        ws = np.sign(ws)*np.sqrt(ws**2+2*phi)
        y += a*norm.pdf(ws+mu, mu, sigma)
    return y

def fzw(z, w, **theta):
    sigma_w = theta['sigma_w']
    a = theta['a']
    w0 = theta['w0']
    dim = len(a)
    popt = np.zeros(3*dim)
    for i in range(dim):
        popt[3*i] = a[i]
        popt[3*i+1] = w0[i]
        popt[3*i+2] = sigma_w[i]
    phis = phi_mod(z, **theta)
    return n_gaussian_phi(w, phis, *popt)

def fw_un(w, **theta):
    z = np.linspace(-1000, 1000, 1000)

    W, Z = np.meshgrid(w, z)
    FZW = fzw(Z, W, **theta)
    Fw = np.trapz(FZW, z, axis=0)
    return Fw

def fw(w, **theta):
    Fw = fw_un(w, **theta)
    Fw = Fw/np.trapz(Fw, w)
    return Fw