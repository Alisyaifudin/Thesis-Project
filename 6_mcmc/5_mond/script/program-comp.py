import sys
import pathlib
from os.path import abspath, join
from hammer import mond
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
sys.path.append(root_dir)
from utils import Program
# initilization
labels = [r'$\rho_b\times 10^2$', r'$\mu_0$' ,r'$\log \nu_0$', 
          r'$R\times 10^3$', r'$z_{\odot}$', r'$w_0$', 
          r'$\log \sigma_{w}$', r'$q_{\sigma,w}$', r'$\log a$', r'$q_a$']
labs = ['rhob', 'mu0', 'log_nu0', 'R', 'zsun', 'w0', 'log_sigmaw', 'q_sigmaw', 'log_a', 'q_a']

root_data_path = join(root_dir, 'Data')

default_props = {
    'steps0': 1_500,
    'burn0': 500,
    'steps': 22_000,
    'burn': 2000,
    'labels': labels,
    'labs': labs,
    'indexes': [12] + list(range(24, len(labs)+23)),
    'root_path': join(root_data_path, 'MCMC', 'mond', 'comp'),
    'zpath': join(root_data_path, 'MCMC', 'dm_mock', 'mock', 'comp'),
    'wpath': join(root_data_path, 'MCMC', 'dm_mock', 'mock', 'comp'),
}

if __name__ == '__main__':
    app = Program(mond)
    for key, val in default_props.items():
        app.add(key, val)
    app.main()
    

