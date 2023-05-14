import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dddm')
sys.path.append(root_dir)
from utils import plot_corner, style
style()

model = int(sys.argv[1])
data = int(sys.argv[2])

chain = np.load(join(data_dir, 'data', f'chain-{model}-{data}.npy'))

rhob = chain[:, :, :12].sum(axis=2).T/1E-2
rhoDM = chain[:, :, 24].T/1E-2
sigmaDD = chain[:, :, 25].T
log_hDD = chain[:, :, 26].T
rhoD = rhoDM + sigmaDD/(4*np.exp(log_hDD))/1E-2
nu0 = chain[:, :, 27].T
R = chain[:, :, 28].T/1E-3
zsun = chain[:, :, 29].T
w0 = chain[:, :, 30].T
log_sigmaw1 = chain[:, :, 31].T
log_a1 = chain[:, :, 32].T

params = [rhob, rhoD, rhoDM, sigmaDD, log_hDD, nu0, R, zsun, w0, log_sigmaw1, log_a1]
labels = [r'$\rho_b\times 10^2$', r'$\rho_{\textup{D}}\times 10^2$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\sigma_{\textup{DD}}$', r'$h_{\textup{DD}}$', r'$\log \nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
if model == 2:
    log_sigmaw2 = chain[:, :, 33].T
    log_a2 = chain[:, :, 34].T
    params.append(log_sigmaw2)
    params.append(log_a2)
    labels.append(r'$\log \sigma_{w2}$')
    labels.append(r'$\log a_2$')
params = np.array(params).T

t0 = time()
plot_corner(params, labels, path=join(data_dir, "plots", f"corner-{model}-{data}.png"))
dt = time() - t0
print(f"Time to plot corner: {dt:.2f} s")
