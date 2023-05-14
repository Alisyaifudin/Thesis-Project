import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join, curdir
import sys
from hammer import dddm
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'dddm')
sys.path.append(root_dir)
from glob import glob
import vaex
from utils import calculate_probs
from init import init
from datetime import datetime

model = int(sys.argv[1])
data = int(sys.argv[2])

ndim, nwalkers = init(model)

chain = np.load(join(data_dir, 'data', f'chain-{model}-{data}.npy'))

locs = np.load(join(data_dir, 'data', f'locs-{model}.npy'))
scales = np.load(join(data_dir, 'data', f'scales-{model}.npy'))

zfiles = glob(join(root_dir, 'Data', 'Effective-Volume-v', '*.hdf5'))
zfiles.sort()
wfiles = glob(join(root_dir, 'Data', 'Velocity-Distribution-v', 'gaia*.hdf5'))
wfiles.sort()
zfiles, wfiles

zfile = zfiles[data]
wfile = wfiles[data]
zdata = vaex.open(zfile)

zmid = zdata['z'].to_numpy()
znum = zdata['znum'].to_numpy()
zerr = zdata['err'].to_numpy()

wdata = vaex.open(wfile)

wmid = wdata['w'].to_numpy()
wnum = wdata['num'].to_numpy()
werr = wdata['err'].to_numpy()

zdata = (zmid, znum, zerr)
wdata = (wmid, wnum, werr)

probs = calculate_probs(dddm, chain, ndim, zdata, wdata, locs, scales, batch=1000)
likelihood = probs[:, 2]
max_likelihood = np.max(likelihood)

# calculate BIC
bic = -2 * max_likelihood + ndim * np.log(3*len(zmid)+3*len(wmid))
print(f"BIC: {bic}")
with open(join(data_dir, 'bic.txt'), 'a') as f:
    f.write(f"{model},{data},{bic},{datetime.now()}\n")