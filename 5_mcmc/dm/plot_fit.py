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
data_dir = join(root_dir, 'Data', 'MCMC', 'dm')
from utils import plot_fit, style 
from init import init
style()

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

plot_fit(dm, zdata, wdata, chain, ndim, path=join(data_dir, 'plots', f'fit-{model}-{data}.png'))