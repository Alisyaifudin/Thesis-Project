import sys
import pathlib
from os.path import abspath, join
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..', '..'))
root_data_path = join(root_dir, 'Data')
sys.path.append(root_dir)egg
from utils import Program, Model

default_props = {
    'burn': 1000,
    'step0': 500,
    'step': 10_000,
    'it': 5,
    'thin': 100,
    'm': 10,
    'log': True,
    'result_path': join(root_data_path, 'MCMC-no', 'no', 'mock'),
    'z_dir_path': join(root_data_path, 'MCMC-no', 'mock', 'data', 'mock'),
    'w_dir_path': join(root_data_path, 'MCMC-no', 'mock', 'data', 'mock'),
    'alpha': 0.01,
    'model': Model.NO,
    'nsample': 50_000,
}

if __name__ == '__main__':
    app = Program()
    for key, val in default_props.items():
        app.add(key, val)
    app.main()