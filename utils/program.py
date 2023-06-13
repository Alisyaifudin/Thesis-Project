from datetime import datetime
import sys
import pathlib
from os.path import abspath, join
from time import time
import argparse
import numpy as np 
import sys
current = pathlib.Path(__file__).parent.resolve()
root_dir = abspath(join(current, '..'))
sys.path.append(root_dir)
import utils
# initilization
parser = argparse.ArgumentParser(description='Run mcmc in seperate process')


def validate_args(args):
    i = args.data
    try:
        i = int(i)
    except:
        raise ValueError('data must be integer')
    return i


def timestamp_decorator(func):
    def wrapper(*args, **kwargs):
        t0 = time()
        i = validate_args(args[1])
        print(f"[{datetime.now()}] Starting {func.__name__} for index {i}")
        result = func(*args, **kwargs)
        print(
            f"[{datetime.now()}] Finished {func.__name__} in {np.round(time() - t0, 2)} seconds")
        return result
    return wrapper


default_props = {
    'step0': 1_500,
    'step': 20_000,
    'it': 2,
    'thin': 100,
    'labels': None,
    'labs': None,
    'indexes': None,
    'root_path': None,
    'z_dir_path': None,
    'phi_dir_path': None,
    'alpha': 0.01,
    'model': None
}

models = ['DM', 'DDDM', 'no']

class Program:
    def __init__(self, func):
        self._ready = False
        self.props = default_props
        self.func = func
        print('Program initilized')

    def add(self, key, val):
        if key == "model" and not val in models:
            raise ValueError("model must be 'DM', 'DDDM', or 'no'")  
        self.props[key] = val

    def ready(self):
        return None not in self.props.values()

    @timestamp_decorator
    def run_mcmc(self, args):
        i = validate_args(args)
        z_dir_path = self.props['z_dir_path']
        phi_dir_path = self.props['phi_dir_path']
        print(f'\tLoading data for from')
        print(f'\tz_dir_path: {z_dir_path}')
        print(f'\tz_dir_path: {phi_dir_path}')
        zdata = utils.get_data(z_dir_path, i, "z")
        output_path = join(self.props['root_path'],
                           'data', f'chain-{i:02d}.npy')
        result = (utils.mcmc() 
                          .index(i)
                          .z_dir_path(z_dir_path)
                          .phi_dir_path(phi_dir_path)
                          .model("DM")
                          .step0(self.props['step0'])
                          .step(self.props['step'])
                          .thin(self.props['thin'])
                          .run(self.props['it'])
                 )
        np.save(output_path, result['chain'])
        print(f'\tChain saved to {output_path}')

    @timestamp_decorator
    def plot_chain(self, args):
        i = validate_args(args)
        chain_path = join(self.props['root_path'],
                          'data', f'chain-{i:02d}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['root_path'],
                           'plots', f'chain-{i:02d}.pdf')
        params = utils.get_params(chain, self.props['indexes'], self.props['labs'])
        (utils.plot_chain()
          .params(params)
          .labels(self.props['labels'])
          .alpha(self.props['alpha'])
          .path(output_path)
          .run()
        )
        print(f'\tChain plot saved to {output_path}')

    @timestamp_decorator
    def plot_corner(self, args):
        i = validate_args(args)
        chain_path = join(self.props['root_path'],
                          'data', f'chain-{i:02d}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_path = join(self.props['root_path'],
                           'plots', f'corner-{i:02d}.pdf')
        params = utils.get_params(chain, self.props['indexes'], self.props['labs'])
        (utils.plot_corner() 
          .params(params) 
          .labels(self.props['labels']) 
          .path(output_path)
          .run()
        )
        print(f'\tCorner plot saved to {output_path}')

    @timestamp_decorator
    def plot_fit(self, args):
        i = validate_args(args)
        chain_path = join(self.props['root_path'],
                          'data', f'chain-{i:02d}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        ndim = chain.shape[2]
        flat_sample = chain.reshape((-1, ndim))
        zdata = utils.get_data(self.props['z_dir_path'], i, "z")
        zmid = zdata[0]
        zmax = np.max(np.abs(zmid))*2
        output_path = join(self.props['root_path'],
                           'plots', f'fit-{i:02d}.pdf')
        (utils.plot_fit_z()
          .index(i)
          .z_dir_path(self.props['z_dir_path'])
          .phi_dir_path(self.props['phi_dir_path']) 
          .flat(flat_sample)
          .zmax(zmax)
          .model(self.props['model'])
          .path(output_path)
          .run()
        )
        print(f'\tFit plot saved to {output_path}')

    @timestamp_decorator
    def calculate_prob(self, args):
        i = validate_args(args)
        chain_path = join(self.props['root_path'],
                          'data', f'chain-{i:02d}.npy')
        chain = np.load(chain_path)
        print(f'\tLoading chain from\n\t{chain_path}')
        output_file = join(self.props['root_path'], f'stats.txt')
        ndim = chain.shape[2]
        flat_sample = chain.reshape((-1, ndim))
        (utils.calculate_prob()
          .index(i)
          .z_dir_path(self.props['z_dir_path'])
          .phi_dir_path(self.props['phi_dir_path'])
          .model(self.props['model'])
          .flat(flat_sample) 
          .path(output_file) 
          .run()
        )
        print(f'\tProbabilities saved to {output_file}')

    @timestamp_decorator
    def all(self, args):
        self.run_mcmc(args)
        self.plot_chain(args)
        self.plot_corner(args)
        self.plot_fit(args)
        self.calculate_prob(args)

    def main(self):
        if not self.ready():
            for k, v in self.props.items():
                if v is None:
                    print(f'{k} is not set')
            raise ValueError('Program not ready yet')
        parser = argparse.ArgumentParser(
            description='Run MCMC in seperate process')

        parser.add_argument(
            '-d', '--data', help='data index (check data folder)', required=True)
        subparsers = parser.add_subparsers(
            title='Subcommands', dest='subcommand')

        run_mcmc_parser = subparsers.add_parser(
            'run_mcmc', help='run_mcmc program')
        run_mcmc_parser.set_defaults(func=self.run_mcmc)

        plot_chain_parser = subparsers.add_parser(
            'plot_chain', help='plot_chain program')
        plot_chain_parser.set_defaults(func=self.plot_chain)

        plot_corner_parser = subparsers.add_parser(
            'plot_corner', help='plot_corner program')
        plot_corner_parser.set_defaults(func=self.plot_corner)

        plot_fit_parser = subparsers.add_parser(
            'plot_fit', help='plot_fit program')
        plot_fit_parser.set_defaults(func=self.plot_fit)

        calculate_prob_parser = subparsers.add_parser(
            'calculate_prob', help='calculate_prob program')
        calculate_prob_parser.set_defaults(func=self.calculate_prob)

        all_parser = subparsers.add_parser(
            'all', help='all program')
        all_parser.set_defaults(func=self.all)

        args = parser.parse_args()
        args.func(args)
