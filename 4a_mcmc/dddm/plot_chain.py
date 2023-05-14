import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dddm')
sys.path.append(root_dir)
from utils import  plot_chain, style
style()

model = int(sys.argv[1])
data = int(sys.argv[2])

chain = np.load(join(data_dir, 'data', f'chain0-{model}-{data}.npy'))
print(chain.shape)

rhob = chain[:, :, :12].sum(axis=2).T
rhoDM = chain[:, :, 24].T
sigmaDD = chain[:, :, 25].T
log_hDD = chain[:, :, 26].T
nu0 = chain[:, :, 27].T
R = chain[:, :, 28].T
zsun = chain[:, :, 29].T
w0 = chain[:, :, 30].T
log_sigmaw1 = chain[:, :, 31].T
log_a1 = chain[:, :, 32].T

params = [rhob, rhoDM, sigmaDD, log_hDD, nu0, R, zsun, w0, log_sigmaw1, log_a1]
labels = [r'$\rho_b$', r'$\rho_{\textup{DM}}$', r'$\sigma_{\textup{DD}}$', r'$h_{\textup{DD}}$', r'$\log \nu_0$', r'$R$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
if model == 2:
    log_sigmaw2 = chain[:, :, 33].T
    log_a2 = chain[:, :, 34].T
    params.append(log_sigmaw2)
    params.append(log_a2)
    labels.append(r'$\log \sigma_{w2}$')
    labels.append(r'$\log a_2$')
params = np.array(params).T
    
print(params.shape)

t0 = time()
plot_chain(params, labels, figsize=(10,10), path=join(data_dir, "plots", f"chain0-{model}-{data}.png"))
dt = time() - t0
print(f"Time to plot chain: {dt:.2f} s")