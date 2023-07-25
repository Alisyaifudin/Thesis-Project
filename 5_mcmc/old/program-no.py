import sys
import pathlib
from os.path import abspath, join
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..', '..'))
root_data_path = join(root_dir, 'Data')
sys.path.append(root_dir)
from utils import Program, Model

default_props = {
    'burn': 500,
    'step0': 500,
    'step': 20_000,
    'it': 3,
    'thin': 20,
    'm': 10,
    'log': True,
    'result_path': join(root_data_path, 'MCMC-no', 'no'),
    'z_dir_path': join(root_data_path, 'Effective-Volume-no'),
    'w_dir_path': join(root_data_path, 'Velocity-Distribution-no'),
    'alpha': 0.01,
    'model': Model.NO,
    'nsample': 50_000,
}

if __name__ == '__main__':
    app = Program()
    for key, val in default_props.items():
        app.add(key, val)
    app.main()