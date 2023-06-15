from matplotlib import pyplot as plt
import numpy as np
from hammer import Model
from os.path import join, abspath, pardir
import sys
import vaex
root_dir = abspath(join(pardir,pardir))
if sys.path is not root_dir:
    sys.path.append(root_dir)

from utils import concat, style
style()
root_data_dir = join(root_dir, 'Data')
baryon_dir = join(root_data_dir, "Baryon")
# load baryons components
df_baryon = vaex.open(join(baryon_dir, "baryon.hdf5"))

# Baryonic density, check table 1 from this https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.081101
rhob = df_baryon["rho"].to_numpy()  # Msun/pc^3
sigmaz = df_baryon["sigma_z"].to_numpy() # km/s
# dark matter density
rhoDM = 0.016
# normalisation of vertical density profile
log_nu0 = 0
# the rotation curve term
R = 3.4E-3
# the solar offset
zsun = 30

w0 = -7
sigmaw1 = 5
sigmaw2 = 10
log_sigmaw = np.log(sigmaw1)
q_sigmaw = sigmaw1/sigmaw2
a1 = 1
a2 = 0.1
log_a = np.log(a1)
q_a = a2/a1
log_phi_b = 5

theta = concat(rhob, sigmaz, rhoDM, log_nu0, R, zsun, w0, log_sigmaw, q_sigmaw, log_a, q_a, log_phi_b)

# integration limits
z_start = 0
z_end = 200

dz = 1
z, phi, Kz = Model.DM.solve_potential(theta, z_start, z_end, dz)