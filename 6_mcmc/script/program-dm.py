import sys
import pathlib
from os.path import abspath, join
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..'))
root_data_path = join(root_dir, 'Data')
sys.path.append(root_dir)
from utils import Program, Model

default_props = {
    'burn': 100,
    'step0': 100,
    'step': 500,
    'it': 1,
    'thin': 5,
    'mask': range(0, 60_000, 20),
    'm': 10,
    'log': True,
    'result_path': join(root_data_path, 'MCMC-no', 'dm'),
    'z_dir_path': join(root_data_path, 'Effective-Volume-no'),
    'vel_dir_path': join(root_data_path, 'MCMC-no', 'vel', 'data'),
    'alpha': 0.01,
    'model': Model.DM,
    'nsample': 50_000,
}

if __name__ == '__main__':
    app = Program()
    for key, val in default_props.items():
        app.add(key, val)
    app.main()