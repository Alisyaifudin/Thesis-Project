import numpy as np
from os.path import abspath, join, pardir
import sys 
from glob import glob 
from hammer import Model
current = abspath("")
root_dir = abspath(join(current, pardir))
root_data_dir = join(root_dir, "Data")
import sys
if not root_dir in sys.path: 
    sys.path.append(root_dir)

from utils import (get_data_z, get_data_w, generate_init)

z_dir_path = join(root_data_dir, 'MCMC-mock', 'z')
w_dir_path = join(root_data_dir, 'MCMC-mock', 'z')

index = 0
z_files = glob(join(z_dir_path, "z*"))
z_files.sort()
w_files = glob(join(z_dir_path, "w*"))
w_files.sort()

name = z_files[index].split("/")[-1].replace(".hdf5", "").replace("z_", "")
zdata_ori = get_data_z(z_files[index])
wdata = get_data_w(w_files[index])
zmid, znum, comp = zdata_ori
# mask = np.abs(zmid) < 200
zmid = zmid[:1]
znum = znum[:1]
comp = comp[:1]

zdata = (zmid, znum, comp)

log_nu0_max = np.log(zdata[1].max())
log_a_max = np.log(wdata[1].max())
model = Model.DM
init = generate_init(model, log_nu0_max, log_a_max)
locs = init['locs']
scales = init['scales']    
indexes = init['indexes']
labs = init['labs']
labels = init['labels']

ndim = len(locs)
nwalker = 10*ndim
p0 = model.generate_p0(nwalker, locs, scales)
prob = model.log_prob_par(p0, zdata, wdata, locs, scales)
print(prob)