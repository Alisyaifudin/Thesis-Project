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
from utils import (get_data, get_params, run_mcmc as mcmc,
                   run_calculate_bic_aic, plot_chain as pchain,
                   plot_corner as pcorner, plot_fit as pfit)
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
    'steps0': 1_500,
    'steps': 20_000,   
    'labels': None,
    'labs': None,
    'indexes': None,
    'root_path': None,
    'zpath': None,
    'wpath': None,
}


class Program:
    def __init__(self, func):
        self._ready = False
        self.props = default_props
        self.func = func
        print('Program initilized')
    def add(self, key, val):
        self.props[key] = val
        
    def ready(self):
        return None not in self.props.values()

    @timestamp_decorator
    def run_mcmc(self, args):
        i = validate_args(args)
        zpath = self.props['zpath']
        wpath = self.props['wpath']
        data = get_data(zpath, wpath, i)
        output_path = join(self.props['root_path'], 'data')
        mcmc(self.func, self.props['labs'], self.props['indexes'], 
             data, i, steps0=self.props['steps0'],
             steps=self.props['steps'], output_path=output_path)


    @timestamp_decorator
    def plot_chain(self, args):
        i = validate_args(args)
        chain = np.load(join(self.props['root_path'], 'data', f'chain-2-{i}.npy'))
        output_path = join(self.props['root_path'], 'plots')
        params = get_params(chain, self.props['indexes'], self.props['labs'])
        pchain(params, self.props['labels'], figsize=(10, 10),
            path=join(output_path, f'chain-2-{i}.pdf'))


    @timestamp_decorator
    def plot_corner(self, args):
        i = validate_args(args)
        chain = np.load(join(self.props['root_path'], 'data', f'chain-2-{i}.npy'))
        output_path = join(self.props['root_path'], 'plots')
        params = get_params(chain, self.props['indexes'], self.props['labs'])
        pcorner(params, self.props['labels'], path=join(output_path, f'corner-2-{i}.pdf'))


    @timestamp_decorator
    def plot_fit(self,args):
        i = validate_args(args)
        chain = np.load(join(self.props['root_path'], 'data', f'chain-2-{i}.npy'))
        zpath = self.props['zpath']
        wpath = self.props['wpath']
        data = get_data(zpath, wpath, i)
        ndim = chain.shape[2]
        zdata, wdata = data
        output_path = join(self.props['root_path'], 'plots')
        pfit(self.func, zdata, wdata, chain, ndim, path=join(output_path, f'fit-2-{i}.pdf'))


    @timestamp_decorator
    def calculate_prob(self, args):
        i = validate_args(args)
        chain = np.load(join(self.props['root_path'], 'data', f'chain-2-{i}.npy'))
        output_file = join(self.props['root_path'], f'stats.txt')
        zpath = self.props['zpath']
        wpath = self.props['wpath']
        data = get_data(zpath, wpath, i)
        run_calculate_bic_aic(self.func, self.props['labs'], data, i, chain, output_file)


    @timestamp_decorator
    def all(self, args):
        self.run_mcmc(args)
        self.plot_chain(args)
        self.plot_corner(args)
        self.plot_fit(args)
        self.calculate_prob(args)


    def main(self):
        if not self.ready():
            for k,v in self.props.items():
                if v is None:
                    print(f'{k} is not set')
            raise ValueError('Program not ready yet')
        parser = argparse.ArgumentParser(
            description='Run MCMC in seperate process')

        parser.add_argument(
            '-d', '--data', help='data index (check data folder)', required=True)
        subparsers = parser.add_subparsers(title='Subcommands', dest='subcommand')

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
        all_parser.set_defaults(func=all)

        args = parser.parse_args()
        args.func(args)
