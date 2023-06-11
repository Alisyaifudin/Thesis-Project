import vaex
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from os.path import join, abspath
from os import pardir 
from scipy import interpolate
from glob import glob
import sys
from scipy.stats import norm
root_dir = abspath(join(pardir, pardir))
sys.path.append(root_dir)
from utils import safe_mkdir
root_data_dir = abspath(join(root_dir, "Data"))
spectral_dir = join(root_data_dir, "Spectral-Class-Cluster-no")
data_dir = join(root_data_dir, "Vertical-Number-no")
safe_mkdir(data_dir)

files = glob(join(spectral_dir, "*.hdf5"))
files.sort()
print(data_dir)
def z_pdf_s(z, b, parallax, parallax_error):
    if b*z <= 0:
        return 0
    return norm.pdf(np.sin(b)/z, loc=parallax, scale=parallax_error)*z**2/parallax_error/np.sin(np.abs(b))**3
z_pdf = np.vectorize(z_pdf_s)

for file in files:
    name = file.split("/")[-1].replace(".hdf5", "")
    gaia = vaex.open(file)
    gaia = gaia[['parallax', 'e_parallax', 'GLAT']]
    gaia = gaia.dropna()
    p = gaia['parallax'].to_numpy()/1000
    p_err = gaia['e_parallax'].to_numpy()/1000
    b = gaia['GLAT'].to_numpy()
    f = p_err/p

    M = 10000
    zs = np.empty((len(p), M))
    for i, (p_i, p_err_i, b_i) in enumerate(zip(p, p_err, tqdm(b, desc=name))):
        f = p_err_i/p_i
        mode = 500
        delta = 100
        if f < 0.05:
            mode = 1/p_i*np.sin(np.abs(b_i))
            delta = p_err_i/p_i*mode
        z = np.linspace(mode-5*delta, mode+5*delta, 1000)
        if b_i < 0:
            z = -z
        pdf = z_pdf(z, b_i, p_i, p_err_i)
        pdf = pdf/pdf.max()
        mask = (pdf>1e-6)
        z = z[mask]
        pdf = pdf[mask]
        z = np.linspace(z.min(), z.max(), 1000)
        pdf = z_pdf(z, b_i, p_i, p_err_i)
        cdf = np.cumsum(pdf)
        cdf = cdf/cdf.max()
        inverse_cdf = interpolate.interp1d(cdf, z, fill_value='extrapolate', bounds_error=False)
        zs_i = inverse_cdf(np.random.rand(M))
        zs[i] = zs_i
    np.save(join(data_dir, name+".npy"), zs)