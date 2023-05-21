import sys
import pathlib
from os.path import abspath, join
from hammer import dddm
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
sys.path.append(root_dir)
from utils import Program
# initilization
labels = [r'$\rho_b\times 10^2$', r'$\rho_{\textup{DM}}\times 10^2$', 
          r"$\sigma_{\textup{DD}}$", r"$\log h_{\textup{DD}}$", r'$\log \nu_0$', 
          r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', r'$\log \sigma_{w1}$', 
          r'$\log a_1$', r'$\log \sigma_{w2}$', r'$\log a_2$']
labs = ['rhob', 'rhoDM', 'sigmaDD', 'log_hDD', 'log_nu0', 'R', 'zsun', 'w0', 
        'log_sigmaw1', 'log_a1', 'log_sigmaw2', 'log_a2']
root_data_path = join(root_dir, 'Data')

default_props = {
    'steps0': 1_500,
    'steps': 20_000,   
    'labels': labels,
    'labs': labs,
    'indexes': [12] + list(range(24, len(labs)+23)),
    'root_path': join(root_data_path, 'MCMC', 'dddm', 'comp'),
    'zpath': join(root_data_path, 'MCMC', 'dm_mock', 'mock', 'comp'),
    'wpath': join(root_data_path, 'MCMC', 'dm_mock', 'mock', 'comp'),
}

if __name__ == '__main__':
    app = Program(dddm)
    for key, val in default_props.items():
        app.add(key, val)
    app.main()
    