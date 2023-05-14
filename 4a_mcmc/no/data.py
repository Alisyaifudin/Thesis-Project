from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from time import time
from os.path import abspath, join, curdir
import sys
from hammer import dm
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
current_dir = abspath(curdir)
from utils import plot_corner, plot_chain, plot_fit
from tqdm import tqdm
from init import init

parallel = False
if len(sys.argv) == 2:
    parallel = sys.argv[1] == 'True'

plt.style.use('seaborn-v0_8-deep') # I personally prefer seaborn for the graph style, but you may choose whichever you want.
params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "text.usetex" : True,
          "font.family" : "serif",
          "font.serif" : ["Computer Modern"]}
plt.rcParams.update(params)

zdata = np.loadtxt(join(root_dir, 'dm', 'data', 'z1.csv'), skiprows=1, delimiter=',')
zmid = zdata[:, 0]
znum = zdata[:, 1]
zerr = zdata[:, 2]

wdata = np.loadtxt(join(root_dir, 'dm', 'data', 'w1.csv'), skiprows=1, delimiter=',')
wmid = wdata[:, 0]
wnum = wdata[:, 1]
werr = wdata[:, 2]

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

ndim = 31
nwalkers = 2*ndim+2

locs = dict(
    rhoDM=-0.02,
    log_nu0=0.5,
    R=3.4E-3,
    zsun=-50,
    w0=-10,
    log_sigmaw=0,
    log_a=-1
)

scales = dict(
    rhoDM=0.08,
    log_nu0=1,
    R=0.6E-3,
    zsun=100,
    w0=5,
    log_sigmaw=3,
    log_a=2
)

keys = list(locs.keys())
locs = np.array(list(locs.values()))
scales = np.array(list(scales.values()))

p0 = dm.generate_p0(nwalkers, locs, scales, kind=1)

t0 = time()
chain = dm.mcmc(500, nwalkers, p0, zdata, wdata, locs, scales, dz=1, verbose=True, parallel=parallel)
print(time() - t0, "s")

rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
nu0 = chain[:, :, 25].T
R = chain[:, :, 26].T
zsun = chain[:, :, 27].T
w0 = chain[:, :, 28].T
log_sigmaw = chain[:, :, 29].T
log_a = chain[:, :, 30].T

params = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, log_sigmaw, log_a]).T
print(params.shape)
labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
plot_chain(params, labels, figsize=(10,10), path="chain0.png")

p0_next = chain[-1, :, :]
t0 = time()
chain = dm.mcmc(2000, nwalkers, p0_next, zdata, wdata, locs, scales, dz=1, verbose=True, parallel=parallel)
print(time() - t0, "s")

rhob = chain[:, :, :12].sum(axis=2).T
sigmaz = chain[:, :, 12:24].sum(axis=2).T
rhoDM = chain[:, :, 24].T
nu0 = chain[:, :, 25].T
R = chain[:, :, 26].T
zsun = chain[:, :, 27].T
w0 = chain[:, :, 28].T
log_sigmaw = chain[:, :, 29].T
log_a = chain[:, :, 30].T

params = np.array([rhob, sigmaz, rhoDM, nu0, R, zsun, w0, log_sigmaw, log_a]).T

labels = labels = [r'$\rho_b$', r'$\sigma_z$', r'$\rho_{\textup{DM}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
plot_chain(params, labels, path="chain1.png")

rhob_f = rhob/1E-2
sigmaz_f = sigmaz
rhoDM_f = rhoDM/1E-2
nu0_f = nu0
R_f = R/1E-3
zsun_f = zsun
w0_f = w0
log_sigmaw_f = log_sigmaw
log_a_f = log_a

flat_samples = np.array([rhob_f, sigmaz_f, rhoDM_f, nu0_f, R_f, zsun_f, w0_f, log_sigmaw_f, log_a_f]).T

labels = [r'$\rho_b\times 10^2$', r'$\sigma_z$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
t0 = time()
plot_corner(flat_samples, labels, path="corner.png")
dt = time() - t0
print(f"Time to plot corner: {dt:.2f} s")

# fit
plot_fit(zdata, wdata, chain, ndim, path="fit.png")