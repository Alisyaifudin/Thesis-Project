import sys
import pathlib
from os.path import abspath, join
from mcmc import dm
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
sys.path.append(root_dir)
from utils import Program
# initilization
labels = [r'$\rho_b\times 10^2$', r'$\rho_{\textup{DM}}\times 10^2$', r'$\log \nu_0$', 
          r'$R\times 10^3$', r'$z_{\odot}$']
labs = ['rhob', 'rhoDM', 'log_nu0', 'R', 'zsun']
root_data_path = join(root_dir, 'Data')

default_props = {
    'step0': 300,
    'step': 20_000,
    'it': 5,
    'thin': 200,
    'labels': labels,
    'labs': labs,
    'indexes': [12] + list(range(24, len(labs)+23)),
    'root_path': join(root_data_path, 'MCMC-no', 'mock', 'result', 'z'),
    'z_dir_path': join(root_data_path, 'MCMC-no', 'mock', 'data', 'z'),
    'phi_dir_path': join(root_data_path, 'MCMC-no', 'mock', 'data','z', 'PHI'),
    'alpha': 0.01,
    'model': "DM"
}

if __name__ == '__main__':
    app = Program(dm)
    for key, val in default_props.items():
        app.add(key, val)
    app.main()
