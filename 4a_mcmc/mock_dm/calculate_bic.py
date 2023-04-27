import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join
from glob import glob
import vaex
import sys
from hammer import dm
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
data_dir = join(root_dir, 'Data', 'MCMC', 'dm_mock')
from utils import calculate_probs
from init import init
from datetime import datetime

tipe = sys.argv[1]
if not tipe in ['z', 'n']:
    raise ValueError('tipe must be "z" or "n"')
model = int(sys.argv[2])
data = int(sys.argv[3])

ndim, nwalkers = init(model)

chain = np.load(join(data_dir, 'data', tipe, f'chain-{model}-{data}.npy'))

locs = np.load(join(data_dir, 'data', tipe, f'locs-{model}.npy'))
scales = np.load(join(data_dir, 'data', tipe, f'scales-{model}.npy'))

zfiles = glob(join(data_dir, 'mock', tipe, 'z*.hdf5'))
zfiles.sort()
wfiles = glob(join(data_dir, 'mock', tipe, 'w*.hdf5'))
wfiles.sort()

zfile = zfiles[data]
wfile = wfiles[data]
zdata = vaex.open(zfile)

zmid = zdata['z'].to_numpy()
znum = zdata['num'].to_numpy()
zerr = zdata['err'].to_numpy()

wdata = vaex.open(wfile)

wmid = wdata['w'].to_numpy()
wnum = wdata['num'].to_numpy()
werr = wdata['err'].to_numpy()

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

probs = calculate_probs(dm, chain, ndim, zdata, wdata, locs, scales, batch=10000)
likelihood = probs[:, 1]
max_likelihood = np.max(likelihood)

# calculate BIC
bic = -2 * max_likelihood + ndim * np.log(3*len(zmid)+3*len(wmid))
print(f"BIC: {bic}")
with open(join(data_dir, f'bic-{tipe}.txt'), 'a') as f:
    f.write(f"{model},{data},{bic},{datetime.now()}\n")