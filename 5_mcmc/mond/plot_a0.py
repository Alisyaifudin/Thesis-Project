import warnings
warnings.filterwarnings('ignore')
import numpy as np
from matplotlib import pyplot as plt
from os.path import abspath, join
import sys
import vaex
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'mond')
from init import init

model = int(sys.argv[1])
data = int(sys.argv[2])

ndim, nwalkers = init(model)

chain = np.load(join(data_dir, 'data', f'chain-{model}-{data}.npy'))

def rar(mu, an):
    return an*np.log(1-1/mu)**(-2)

def simple(mu, an):
    return an/4*((2*mu-1)**2-1)

def standard(mu, an):
    return an/2*np.sqrt((2*mu**2-1)**2-1)

from scipy.stats import norm

mu0 = chain[:,:, 24]

sh = mu0.shape
an = norm.rvs(loc=2.2, scale=0.5, size=sh)

a0_rar = rar(mu0, an)
a0_simple = simple(mu0, an)
a0_standard = standard(mu0, an)
df = vaex.from_arrays(rar=a0_rar.flatten(), simple=a0_simple.flatten(), standard=a0_standard.flatten())
a0_rar = a0_rar[~np.isnan(a0_rar)]
a0_simple = a0_simple[~np.isnan(a0_simple)]
a0_standard = a0_standard[~np.isnan(a0_standard)]


# plot histogram of a0 in one canvas
print("median:", np.median(a0_rar.flatten()), np.median(a0_simple.flatten()), np.median(a0_standard.flatten()))
print("mean:", np.mean(a0_rar.flatten()), np.mean(a0_simple.flatten()), np.mean(a0_standard.flatten()))
plt.figure(figsize=(10, 4))
plt.hist(a0_rar.flatten(), bins=50, alpha=0.5, label='RAR', density=True)
plt.hist(a0_simple.flatten(), bins=50, alpha=0.5, label='simple', density=True)
plt.hist(a0_standard.flatten(), bins=50, alpha=0.5, label='standard', density=True)

plt.xlabel(r'$a_0$')
plt.legend()
plt.savefig(join(data_dir, 'plots', f'a0-{model}-{data}.png'), dpi=100)
df.export(join(data_dir, 'data', f'a0-{model}-{data}.hdf5'), progress=True)
