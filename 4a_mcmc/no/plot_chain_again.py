import warnings
warnings.filterwarnings('ignore')
import numpy as np
from time import time
from os.path import abspath, join
import sys
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
data_dir = join(root_dir, 'Data', 'MCMC', 'no')
from utils import  plot_chain, style
style() 

model = int(sys.argv[1])
data = int(sys.argv[2])

chain = np.load(join(data_dir, 'data', f'chain-{model}-{data}.npy'))

print(chain.shape)

rhob = chain[:, :, :12].sum(axis=2).T
nu0 = chain[:, :, 24].T
R = chain[:, :, 25].T
zsun = chain[:, :, 26].T
w0 = chain[:, :, 27].T
log_sigmaw1 = chain[:, :, 28].T
log_a1 = chain[:, :, 29].T

params = [rhob, nu0, R, zsun, w0, log_sigmaw1, log_a1]
labels = [r'$\rho_{b}\times 10^2$', r'$\nu_0$', r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_w$', r'$\log a$']
if model == 2:
    log_sigmaw2 = chain[:, :, 30].T
    log_a2 = chain[:, :, 31].T
    params.append(log_sigmaw2)
    params.append(log_a2)
    labels.append(r'$\log \sigma_{w2}$')
    labels.append(r'$\log a_2$')
params = np.array(params).T
    
t0 = time()
plot_chain(params, labels, figsize=(10,10), path=join(data_dir, "plots", f"chain-{model}-{data}.png"))
dt = time() - t0
print(f"Time to plot chain: {dt:.2f} s")