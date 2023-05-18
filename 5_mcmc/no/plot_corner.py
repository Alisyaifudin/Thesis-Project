import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
data_dir = join(root_dir, 'Data', 'MCMC', 'no')
from utils import plot_corner, style
from scipy.stats import norm
style()

rhob_locs = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018, 0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015,
]
rhob_scales= [
    0.00312, 0.00554, 0.00070, 0.00003, 0.00006, 0.00018, 0.00018, 0.00029, 0.00072, 0.00280,
    0.00100, 0.00050,
]

model = int(sys.argv[1])
data = int(sys.argv[2])

chain = np.load(join(data_dir, 'data', f'chain-{model}-{data}.npy'))

sh = chain.shape
rhob = chain[:, :, :12].sum(axis=2).T/1E-2
rhob0 = np.empty((sh[0], sh[1], 12))
for i in range(12):
    rhob0[:, :, i] = norm.rvs(loc=rhob_locs[i], scale=rhob_scales[i], size=(sh[0], sh[1]))

rhob0 = rhob0.sum(axis=2).T/1E-2
rhoD = rhob - rhob0
nu0 = chain[:, :, 24].T
R = chain[:, :, 25].T
zsun = chain[:, :, 26].T
w0 = chain[:, :, 27].T
log_sigmaw1 = chain[:, :, 28].T
log_a1 = chain[:, :, 29].T

params = [rhob0, rhoD, rhob, nu0, R, zsun, w0, log_sigmaw1, log_a1]
labels = [r'$\rho_{b0}\times 10^2$', r'$\rho_{\textup{D}}\times 10^2$', r'$\rho_{b}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
if model == 2:
    log_sigmaw2 = chain[:, :, 30].T
    log_a2 = chain[:, :, 31].T
    params.append(log_sigmaw2)
    params.append(log_a2)
    labels.append(r'$\log \sigma_{w2}$')
    labels.append(r'$\log a_2$')
params = np.array(params).T

t0 = time()
plot_corner(params, labels, path=join(data_dir, "plots", f"corner-{model}-{data}.png"))
dt = time() - t0
print(f"Time to plot corner: {dt:.2f} s")
