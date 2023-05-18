import warnings
warnings.filterwarnings('ignore')
import vaex
import numpy as np
from glob import glob
from os.path import abspath, join
import sys
from hammer import no
root_dir = abspath(join('..', '..'))
data_dir = join(root_dir, 'Data', 'MCMC', 'no')
sys.path.append(root_dir)
from utils import style, calculate_probs
from init import init
from datetime import datetime
style()

parallel = False
model = int(sys.argv[1])
data = int(sys.argv[2])

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

ndim, nwalkers = init(model)

locs = np.load(join(data_dir, 'data', f'locs-{model}.npy'))
scales = np.load(join(data_dir, 'data', f'scales-{model}.npy'))
chain = np.load(join(data_dir, 'data', f'chain0-{model}-{data}.npy'))

probs = calculate_probs(no, chain, ndim, zdata, wdata, locs, scales, batch=10000)
likelihood = probs[:, 1]
max_likelihood = np.max(likelihood)

# calculate BIC
bic = -2 * max_likelihood + ndim * np.log(3*len(zmid)+3*len(wmid))
print(f"BIC: {bic}")
with open(join(data_dir, 'bic.txt'), 'a') as f:
    f.write(f"{model},{data},{bic},{datetime.now()}\n")