import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join, curdir
import sys
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
data_dir = join(root_dir, 'Data', 'MCMC', 'dm_mock')
from utils import plot_corner, style
style()

tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be "z" or "n"')
model = int(sys.argv[2])
data = int(sys.argv[3])

chain = np.load(join(data_dir, 'data', tipe, f'chain-{model}-{data}.npy'))

rhob = chain[:, :, :12].sum(axis=2).T/1E-2
rhoDM = chain[:, :, 24].T/1E-2
nu0 = chain[:, :, 25].T
R = chain[:, :, 26].T/1E-3
zsun = chain[:, :, 27].T
w0 = chain[:, :, 28].T
log_sigmaw1 = chain[:, :, 29].T
log_a1 = chain[:, :, 30].T
params = [rhob,  rhoDM, nu0, R, zsun, w0, log_sigmaw1, log_a1]
labels = [r'$\rho_b\times 10^2$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\log \nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', r'$\log a_1$']
if model == 2:
    log_sigmaw2 = chain[:, :, 31].T
    log_a2 = chain[:, :, 32].T
    params.append(log_sigmaw2)
    params.append(log_a2)
    labels.append(r'$\log \sigma_{w2}$')
    labels.append(r'$\log a_2$')
params = np.array(params).T

t0 = time()
plot_corner(params, labels, path=join(data_dir, "plots", tipe, f"corner-{model}-{data}.png"))
dt = time() - t0
print(f"Time to plot corner: {dt:.2f} s")
