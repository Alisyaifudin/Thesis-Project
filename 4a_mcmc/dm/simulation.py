import numpy as np
from matplotlib import pyplot as plt
from hammer import dm
from time import time
from os.path import join, abspath
root_dir = abspath(join('..', '..'))

data_dir = join(root_dir, 'Data', 'MCMC', 'dm', 'simulation')

rhob = [
    0.0104, 0.0277, 0.0073, 0.0005, 0.0006, 0.0018,
    0.0018, 0.0029, 0.0072, 0.0216, 0.0056, 0.0015
]
sigmaz = [
    3.7, 7.1, 22.1, 39.0, 15.5, 7.5, 12.0, 
    18.0, 18.5, 18.5, 20.0, 20.0]
rhoDM = [0.016]
log_nu0 = [1]
R = [3.4E-3]
zsun = [20]
w0 = [-7.]
log_sigmaw = [np.log(5.)]
log_a = [np.log(1.)]

theta = np.array([rhob + sigmaz + rhoDM + log_nu0 + R + zsun+w0 + log_sigmaw + log_a]).flatten()

C_pc_Myr = 3.086E13 / (1E6 * 365.25 * 24 * 60 * 60)

N = 1000000
z0 = np.random.rand(N)*300-150 # pc
n1 = np.floor(N*0.6).astype(int)
n2 = N - n1
vz01 = np.random.randn(n1)*6
vz02 = np.random.randn(n2)*20
vz0 = np.concatenate([vz01, vz02])
# # convert vz0 from km/s to pc/Myr
vz0 = vz0 / C_pc_Myr

dz = 1
z_start = 0
z_end = 2000
dt = 1
t_end = 2000 

t0 = time()
sol = dm.run_simulation(theta, z_start, z_end, dz, z0, vz0, dt, t_end, parallel=True, batch=10000)
t1 = time()
print('Simulation took {:.2f} seconds'.format(t1-t0))

np.save(join(data_dir, 'sol.npy'), sol)