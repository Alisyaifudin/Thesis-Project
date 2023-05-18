import warnings
warnings.filterwarnings('ignore')
import numpy as np
from os.path import abspath, join
from glob import glob
import vaex
import sys 
from hammer import dddm
root_dir = abspath(join('..', '..'))
sys.path.append(root_dir)
data_dir = join(root_dir, 'Data', 'MCMC', 'dddm')
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

print(f'Calculating probs for model {model} and data {data} at {datetime.now()}')
probs = calculate_probs(dddm, chain, ndim, zdata, wdata, locs, scales, batch=10000)
print(f'Finished calculating probs for model {model} and data {data} at {datetime.now()}')
print("=====================================================================")
np.save(join(data_dir, 'data', f'probs-{model}-{data}.npy'), probs)