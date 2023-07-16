import vaex
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from os.path import join, abspath
from os import pardir 
from scipy import interpolate
from glob import glob
import sys
from hammer import vertical
from scipy.stats import norm
root_dir = abspath(join(pardir, pardir))
sys.path.append(root_dir)
from utils import safe_mkdir
root_data_dir = abspath(join(root_dir, "Data"))
spectral_dir = join(root_data_dir, "Cluster", "no")
data_dir = join(root_data_dir, "Vertical-Distance")
safe_mkdir(data_dir)
data_dir = join(root_data_dir, "Vertical-Distance", "no")
safe_mkdir(data_dir)

files = glob(join(spectral_dir, "*.hdf5"))
files.sort()

for file in files:
    name = file.split("/")[-1].replace(".hdf5", "")
    gaia = vaex.open(file)
    gaia = gaia[['parallax', 'e_parallax', 'GLAT']]
    gaia = gaia.dropna()
    p = gaia['parallax'].to_numpy()/1000
    p_err = gaia['e_parallax'].to_numpy()/1000
    b = gaia['GLAT'].to_numpy()
    n = 1000
    zs = vertical.generate_z(p[:1000], p_err[:1000], b[:1000], n=n)
    print("halo", zs.shape)